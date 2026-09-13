// Headless bridge to MPCPlanner::Planner.
//
// One long-lived process holding one Planner.  Requests arrive as whitespace
// separated lines on stdin, replies go to stdout, everything else to stderr.
// The planner itself is the upstream one: this file only serialises state and
// calls reset / onDataReceived / solveMPC / getSolution.  No control law of our
// own runs here.
//
// Protocol (one request per line, one reply per line):
//
//   INFO
//     -> INFO <N> <dt> <max_obstacles> <control_frequency> <config_path>
//
//   RESET
//     -> RESET_OK
//
//   STEP x y psi v goal_x goal_y
//        NPATH n  px py  (n times)
//        NOBS  m  id ox oy oangle radius NPRED p  (mx my mangle major minor) x p   (m times)
//     -> STEP <success> <v_cmd> <w_cmd> <solve_ms> <total_ms> <exit_code>
//        success is 1/0.  On failure v_cmd/w_cmd carry the planner's own
//        braking input, computed exactly as the upstream ROS node does.
//
//   QUIT
//     -> BYE
//
// Any malformed request yields "ERR <reason>" and leaves the planner untouched.

#include <mpc_planner/planner.h>
#include <mpc_planner/data_preparation.h>

#include <mpc_planner_solver/state.h>
#include <mpc_planner_types/realtime_data.h>
#include <mpc_planner_util/parameters.h>
#include <mpc_planner_util/load_yaml.hpp>

#include <ros_tools/logging.h>
// Planner holds a unique_ptr<RosTools::Timer>, which planner.h only forward
// declares; the destructor cannot be instantiated here without this.
#include <ros_tools/profiling.h>
// The planner's modules build ROS publishers for their (disabled) visuals as
// soon as they are constructed, so roscpp must be initialised even though the
// bridge itself neither publishes nor subscribes.  A roscore must be running.
#include <ros/ros.h>
#include <mpc_planner_solver/solver_interface.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
#include <fstream>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace MPCPlanner;

namespace
{
    // roscpp writes its INFO stream to stdout, so the protocol channel is moved
    // to a private descriptor and fd 1 is pointed at stderr.  After this no
    // library can corrupt a reply, whatever it prints.
    int g_protocol_fd = -1;

    void claimProtocolChannel()
    {
        g_protocol_fd = dup(STDOUT_FILENO);
        if (g_protocol_fd < 0)
            throw std::runtime_error("cannot_duplicate_stdout");
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
            throw std::runtime_error("cannot_redirect_stdout");
    }

    void reply(const std::string &text)
    {
        const std::string line = text + "\n";
        size_t written = 0;
        while (written < line.size())
        {
            ssize_t n = write(g_protocol_fd, line.data() + written, line.size() - written);
            if (n <= 0)
                return;
            written += (size_t)n;
        }
    }

    // Reads one token, throwing a labelled error rather than silently yielding 0.
    template <typename T>
    T need(std::istringstream &in, const char *what)
    {
        T value;
        if (!(in >> value))
            throw std::runtime_error(std::string("missing_or_bad_") + what);
        return value;
    }

    void expectKeyword(std::istringstream &in, const char *keyword)
    {
        std::string token;
        if (!(in >> token) || token != keyword)
            throw std::runtime_error(std::string("expected_") + keyword);
    }

    double nowMs(const std::chrono::steady_clock::time_point &start)
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    }
}


// ---------------------------------------------------------------------------
// Section 13.4-C: the complete iterate at a failing step.
//
// A failure recorded only as "status 3 at this observation" cannot be replayed:
// the solver is warm-started, so the same observation with a different internal
// state is a different problem.  This writes every stage's primal variables,
// equality multipliers, inequality multipliers and slacks, so the step can be
// reconstructed rather than approximated.
//
// It runs only when MPC_BRIDGE_FAILURE_DUMP names a file, and only on a step
// the solver already reported as failed.  Formal runs leave it unset, so the
// measured path is untouched.
// ---------------------------------------------------------------------------
static void dumpStageField(std::ofstream &out, ocp_nlp_config *config,
                           ocp_nlp_dims *dims, ocp_nlp_out *nlp_out,
                           int stage, const char *field, bool &any_nonfinite,
                           bool comma)
{
    // The size is asked of acados rather than assumed: the constraint count
    // differs between the two module configurations and between stages, and a
    // wrong length here would read past the solver's buffers.
    const int size = ocp_nlp_dims_get_from_attr(config, dims, nlp_out, stage,
                                                field);
    if (comma)
        out << ",\n";
    out << "    \"" << field << "\": ";
    if (size <= 0)
    {
        out << "[]";
        return;
    }
    std::vector<double> buffer(static_cast<size_t>(size), 0.);
    ocp_nlp_out_get(config, dims, nlp_out, stage, const_cast<char *>(field),
                    buffer.data());
    out << "[";
    for (int i = 0; i < size; ++i)
    {
        if (!std::isfinite(buffer[i]))
            any_nonfinite = true;
        out << (i ? "," : "") << std::setprecision(17) << buffer[i];
    }
    out << "]";
}

static void dumpIterate(const std::string &path, const std::string &request,
                        const std::shared_ptr<MPCPlanner::Solver> &solver,
                        int selected_branch)
{
    std::ofstream out(path, std::ios::app);
    if (!out)
        return;
    auto *config = solver->nlpConfigForDiagnostics();
    auto *dims = solver->nlpDimsForDiagnostics();
    auto *nlp_out = solver->nlpOutForDiagnostics();
    const int N = solver->N;
    bool nonfinite = false;

    // Everything this acados version exposes per stage: the primal variables,
    // the algebraic variables, the equality multipliers, the inequality
    // multipliers and both slack sets.  There is no `t` accessor in this
    // version, so the inequality residuals are not dumped and that is stated
    // rather than quietly omitted.
    static const char *FIELDS[] = {"x", "u", "z", "pi", "lam", "sl", "su"};

    out << "{\n  \"request\": \"" << request << "\",\n";
    out << "  \"selected_branch\": " << selected_branch << ",\n";
    out << "  \"solver_id\": " << solver->_solver_id << ",\n";
    out << "  \"N\": " << N << ",\n";
    out << "  \"fields_absent_in_this_acados\": [\"t\"],\n";
    out << "  \"stages\": [\n";
    for (int k = 0; k <= N; ++k)
    {
        out << "   {\n";
        bool comma = false;
        for (const char *field : FIELDS)
        {
            // `u` and `pi` do not exist on the terminal stage.
            if (k == N && (!std::strcmp(field, "u") || !std::strcmp(field, "pi")))
                continue;
            dumpStageField(out, config, dims, nlp_out, k, field, nonfinite, comma);
            comma = true;
        }
        out << "\n   }" << (k < N ? "," : "") << "\n";
    }
    out << "  ],\n  \"iterate_nonfinite\": " << (nonfinite ? "true" : "false")
        << "\n}\n";
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: mpc_bridge <settings.yaml>" << std::endl;
        return 2;
    }
    const std::string config_path = argv[1];

    claimProtocolChannel();

    ros::init(argc, argv, "mpc_bridge", ros::init_options::AnonymousName |
                                        ros::init_options::NoSigintHandler);
    ros::NodeHandle keep_node_alive;

    Configuration::getInstance().initialize(config_path);

    State state;
    RealTimeData data;
    data.robot_area = {Disc(0., CONFIG["robot_radius"].as<double>())};

    auto planner = std::make_unique<Planner>();

    const int N = CONFIG["N"].as<int>();
    const double integrator_step = CONFIG["integrator_step"].as<double>();
    const int max_obstacles = CONFIG["max_obstacles"].as<int>();
    const double control_frequency = CONFIG["control_frequency"].as<double>();
    const double obstacle_radius = CONFIG["obstacle_radius"].as<double>();

    std::cerr << "[bridge] ready: N=" << N << " dt=" << integrator_step
              << " max_obstacles=" << max_obstacles << std::endl;

    const char *dump_env = std::getenv("MPC_BRIDGE_FAILURE_DUMP");
    const std::string failure_dump = dump_env ? dump_env : "";
    if (!failure_dump.empty())
        std::cerr << "[bridge] failing steps -> " << failure_dump << std::endl;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;

        std::istringstream in(line);
        std::string command;
        in >> command;

        try
        {
            if (command == "QUIT")
            {
                reply("BYE");
                break;
            }
            if (command == "INFO")
            {
                {
                    std::ostringstream out;
                    // num_segments decides how far ahead the contouring spline is
                    // tracked; a caller building the reference needs it to make the
                    // tracked window reach at least as far as the horizon travels.
                    const int num_segments =
                        CONFIG["contouring"]["num_segments"].as<int>();
                    out << std::setprecision(12) << "INFO " << N << " " << integrator_step
                        << " " << max_obstacles << " " << control_frequency
                        << " " << num_segments << " " << config_path;
                    reply(out.str());
                }
                continue;
            }
            if (command == "RESET")
            {
                planner->reset(state, data, true);
                reply("RESET_OK");
                continue;
            }
            if (command != "STEP")
            {
                reply("ERR unknown_command_" + command);
                continue;
            }

            const auto wall_start = std::chrono::steady_clock::now();

            const double x = need<double>(in, "x");
            const double y = need<double>(in, "y");
            const double psi = need<double>(in, "psi");
            const double v = need<double>(in, "v");
            const double goal_x = need<double>(in, "goal_x");
            const double goal_y = need<double>(in, "goal_y");

            state.set("x", x);
            state.set("y", y);
            state.set("psi", psi);
            state.set("v", v);

            data.goal(0) = goal_x;
            data.goal(1) = goal_y;
            data.goal_received = true;

            // Reference path.  Sent every step; the planner is told only when it
            // actually changed, matching the upstream node's behaviour.
            expectKeyword(in, "NPATH");
            const int n_path = need<int>(in, "n_path");
            if (n_path < 0)
                throw std::runtime_error("negative_n_path");

            std::vector<double> path_x(n_path), path_y(n_path);
            for (int i = 0; i < n_path; i++)
            {
                path_x[i] = need<double>(in, "path_x");
                path_y[i] = need<double>(in, "path_y");
            }

            bool path_changed = ((int)data.reference_path.x.size() != n_path);
            if (!path_changed)
            {
                for (int i = 0; i < n_path; i++)
                {
                    if (std::abs(data.reference_path.x[i] - path_x[i]) > 1e-9 ||
                        std::abs(data.reference_path.y[i] - path_y[i]) > 1e-9)
                    {
                        path_changed = true;
                        break;
                    }
                }
            }
            if (path_changed && n_path > 0)
            {
                data.reference_path.clear();
                data.reference_path.x = path_x;
                data.reference_path.y = path_y;
                data.reference_path.psi.push_back(0.0);
                planner->onDataReceived(data, "reference_path");
            }

            // Obstacles with their predictions.
            expectKeyword(in, "NOBS");
            const int n_obs = need<int>(in, "n_obs");
            if (n_obs < 0)
                throw std::runtime_error("negative_n_obs");

            data.dynamic_obstacles.clear();
            for (int i = 0; i < n_obs; i++)
            {
                const int id = need<int>(in, "obstacle_id");
                const double ox = need<double>(in, "obstacle_x");
                const double oy = need<double>(in, "obstacle_y");
                const double oangle = need<double>(in, "obstacle_angle");
                double radius = need<double>(in, "obstacle_radius");
                if (radius <= 0.)
                    radius = obstacle_radius;

                data.dynamic_obstacles.emplace_back(
                    id, Eigen::Vector2d(ox, oy), oangle, radius);
                auto &obstacle = data.dynamic_obstacles.back();

                expectKeyword(in, "NPRED");
                const int n_pred = need<int>(in, "n_pred");
                if (n_pred < 0)
                    throw std::runtime_error("negative_n_pred");
                if (n_pred == 0)
                    continue;

                obstacle.prediction = Prediction(PredictionType::GAUSSIAN);
                bool any_uncertainty = false;
                for (int k = 0; k < n_pred; k++)
                {
                    const double mx = need<double>(in, "pred_x");
                    const double my = need<double>(in, "pred_y");
                    const double mangle = need<double>(in, "pred_angle");
                    const double major = need<double>(in, "pred_major");
                    const double minor = need<double>(in, "pred_minor");
                    obstacle.prediction.modes[0].emplace_back(
                        Eigen::Vector2d(mx, my), mangle, major, minor);
                    if (major > 0.)
                        any_uncertainty = true;
                }

                // Same rule the upstream node applies: zero uncertainty, or the
                // probabilistic switch off, means a deterministic prediction.
                if (!any_uncertainty || !CONFIG["probabilistic"]["enable"].as<bool>())
                    obstacle.prediction.type = PredictionType::DETERMINISTIC;
            }

            ensureObstacleSize(data.dynamic_obstacles, state);
            planner->onDataReceived(data, "dynamic obstacles");

            data.planning_start_time = std::chrono::system_clock::now();
            const auto solve_start = std::chrono::steady_clock::now();
            auto output = planner->solveMPC(state, data);
            const double solve_ms = nowMs(solve_start);

            double v_cmd = 0., w_cmd = 0.;
            if (output.success)
            {
                v_cmd = planner->getSolution(1, "v");  // first predicted state
                w_cmd = planner->getSolution(0, "w");  // first input
            }
            else
            {
                // Upstream braking input, reproduced so an infeasible solve is
                // not silently reported as a commanded stop.
                const double deceleration = CONFIG["deceleration_at_infeasible"].as<double>();
                const double dt = 1. / control_frequency;
                v_cmd = std::max(v - deceleration * dt, 0.);
                w_cmd = 0.0;
            }

            // Solver internals travel with every reply.  A bare success flag
            // cannot tell a numerically broken step from an infeasible one, and
            // guessing between them is exactly what must not happen.
            const auto &info = planner->solverForDiagnostics()->_info;
            // The upstream failure flag is `res_eq > 1e-2`, named ACADOS_QP_FAILURE
            // even though the QP itself reports success.  Reading the residuals
            // directly is the only way to tell a converged-but-imprecise step from
            // a genuinely broken one.
            // Read from the info the winning branch left behind, not from the
            // main solver's NLP object: with T-MPC++ that object is never solved,
            // so querying it returns zeros for every step.
            const double res_stat = info.res_stat_at_exit;
            const double res_eq = info.res_eq_at_exit;
            const double res_ineq = info.res_ineq_at_exit;
            const double res_comp = info.res_comp_at_exit;
            {
                std::ostringstream out;
                out << std::setprecision(12) << "STEP " << (output.success ? 1 : 0)
                    << " " << v_cmd << " " << w_cmd << " " << solve_ms << " "
                    << nowMs(wall_start) << " " << (output.success ? 0 : 1)
                    << " sqp_iter=" << info.sqp_iter
                    << " qp_status=" << info.qp_status
                    << " nlp_res=" << info.nlp_res
                    << " kkt=" << info.kkt_norm_inf
                    << " cost=" << info.pobj
                    << " res_stat=" << res_stat
                    << " res_eq=" << res_eq
                    << " res_ineq=" << res_ineq
                    << " res_comp=" << res_comp
                    << " acados_status=" << info.acados_status
                    << " br_total=" << info.branches_total
                    << " br_enabled=" << info.branches_enabled
                    << " br_ok=" << info.branches_succeeded
                    << " br_worst_exit=" << info.branch_worst_exit
                    << " br_status=" << info.branch_acados_status
                    << " br_res_eq=" << info.branch_res_eq
                    << " nf_par=" << info.nonfinite_params
                    << " nf_ws=" << info.nonfinite_warmstart
                    << " nf_first=" << info.first_nonfinite_param
                    << " max_par=" << info.max_abs_param
                    << " hpipm_raw=" << info.hpipm_raw_status
                    << " hpipm_iter=" << info.hpipm_iter
                    << " orig_disabled=" << info.orig_disabled
                    << " orig_exit=" << info.orig_exit
                    << " orig_status=" << info.orig_acados_status
                    << " branch=" << info.selected_branch
                    << " spline0=" << planner->getSolution(0, "spline")
                    << " splineN=" << planner->getSolution(N, "spline")
                    << " vN=" << planner->getSolution(N, "v");
                reply(out.str());
            }

            // Keep the exact request that failed, so it can be replayed against a
            // different configuration instead of being reasoned about.
            if (!output.success && !failure_dump.empty())
            {
                std::ofstream dump(failure_dump, std::ios::app);
                if (dump)
                    dump << "# sqp_iter=" << info.sqp_iter
                         << " qp_status=" << info.qp_status
                         << " hpipm_raw=" << info.hpipm_raw_status
                         << " acados_status=" << info.acados_status
                         << " nlp_res=" << info.nlp_res
                         << " kkt=" << info.kkt_norm_inf << "\n"
                         << line << "\n";
                dumpIterate(failure_dump + ".iterate", line,
                            planner->solverForDiagnostics(),
                            info.selected_branch);
            }
        }
        catch (const std::exception &e)
        {
            reply(std::string("ERR ") + e.what());
            std::cerr << "[bridge] error: " << e.what() << std::endl;
        }
    }

    return 0;
}
