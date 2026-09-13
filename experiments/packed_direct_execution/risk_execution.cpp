// Direct Hermite query experiment only.  Mathematical expression matches the
// mode==0 path at jinglongjiang/bayes-occ-mpc commit 82857b3.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>
#include <sched.h>

// This non-inlined, exported function mirrors the upstream point_query.
extern "C" double point_query(double d,int j,int m,const double* variance,const double* sigma,const double* radius,
    const int64_t* component,const double* table,const double* derivative,int nodes,double spacing) {
    if(variance[j]<1e-9)return d<=radius[m]?1.:0.;
    double z=(d-radius[m])/sigma[j];
    if(z>=8.)return 6.22096057427174e-16;
    if(z < -8.)return 1.;
    double index=(z+8.)/spacing;int left=std::min(std::max(int(std::floor(index)),0),nodes-2);
    double t=std::min(std::max(index-left,0.),1.),t2=t*t,t3=t2*t;
    int64_t slot=component[j]*nodes+left;
    double q=(2*t3-3*t2+1)*table[slot]+(t3-2*t2+t)*spacing*derivative[slot]
        +(-2*t3+3*t2)*table[slot+1]+(t3-t2)*spacing*derivative[slot+1];
    if(!std::isfinite(q))return std::numeric_limits<double>::quiet_NaN();
    return std::max(0.,std::min(1.,q));
}

// Extracted direct-only loop, not a copy of the complete project back end.
extern "C" void reference_direct(const double* positions,int candidates,int horizon,
    const double* centers,const double* variance,const double* sigma,const double* radius,
    const int64_t* component,const double* table,const double* derivative,int nodes,double spacing,
    const double* weights,const int64_t* offsets,const double* existence,int people,
    double* lower,double* upper,int64_t* counts) {
    counts[0]=counts[1]=counts[2]=0;
    for(int n=0;n<candidates;++n) for(int k=0;k<horizon;++k) {
        double hl=0.,hu=0.,x=positions[2*(n*horizon+k)],y=positions[2*(n*horizon+k)+1];
        for(int i=0;i<people;++i) {
            double pl=0.,pu=0.;
            for(int m=offsets[i];m<offsets[i+1];++m) {
                int j=m*horizon+k;double dx=x-centers[2*j],dy=y-centers[2*j+1];
                pl+=weights[m]*point_query(std::sqrt(dx*dx+dy*dy),j,m,variance,sigma,radius,component,table,derivative,nodes,spacing);
                ++counts[0];
            }
            pu=pl;
            if(!std::isfinite(pl) || !std::isfinite(pu)) {
                lower[0]=upper[0]=std::numeric_limits<double>::quiet_NaN();return;
            }
            hl-=std::log1p(-std::min(std::max(pl*existence[i],0.),1.-1e-12));
            hu-=std::log1p(-std::min(std::max(pu*existence[i],0.),1.-1e-12));
        }
        lower[n*horizon+k]=hl;upper[n*horizon+k]=hu;
    }
}

static inline double query_local(double d,double v,double s,double r,
    const double* tab,const double* deriv,int nodes,double spacing) {
    if(v<1e-9)return d<=r?1.:0.;
    double z=(d-r)/s;
    if(z>=8.)return 6.22096057427174e-16;
    if(z < -8.)return 1.;
    double index=(z+8.)/spacing;int left=std::min(std::max(int(std::floor(index)),0),nodes-2);
    double t=std::min(std::max(index-left,0.),1.),t2=t*t,t3=t2*t;
    double q=(2*t3-3*t2+1)*tab[left]+(t3-2*t2+t)*spacing*deriv[left]
        +(-2*t3+3*t2)*tab[left+1]+(t3-t2)*spacing*deriv[left+1];
    if(!std::isfinite(q))return std::numeric_limits<double>::quiet_NaN();
    return std::max(0.,std::min(1.,q));
}

struct Group {
    int begin,end; bool common;
    double v,s,r,existence; int64_t component;
};
struct Context {
    int horizon,modes,people,nodes; double spacing;
    std::vector<double> x,y,variance,sigma,radius,weights,table,derivative;
    std::vector<int64_t> component;std::vector<Group> groups;
    std::vector<int> last_cpus;
    Context(const double* centers,const double* v,const double* s,const double* r,
        const int64_t* c,const double* t,const double* d,int nt,int nn,double spacing_,
        const double* w,const int64_t* offsets,const double* ex,int n,int h)
        :horizon(h),modes(offsets[n]),people(n),nodes(nn),spacing(spacing_),
         x(size_t(h)*modes),y(size_t(h)*modes),variance(size_t(h)*modes),
         sigma(size_t(h)*modes),radius(r,r+modes),weights(w,w+modes),
         table(t,t+size_t(nt)*nn),derivative(d,d+size_t(nt)*nn),
         component(size_t(h)*modes),groups(size_t(h)*n) {
        for(int k=0;k<h;++k) for(int m=0;m<modes;++m) {
            size_t in=size_t(m)*h+k,out=size_t(k)*modes+m;
            x[out]=centers[2*in];y[out]=centers[2*in+1];
            variance[out]=v[in];sigma[out]=s[in];component[out]=c[in];
        }
        for(int k=0;k<h;++k) for(int i=0;i<n;++i) {
            Group g;g.begin=offsets[i];g.end=offsets[i+1];g.existence=ex[i];g.common=true;
            size_t first=size_t(k)*modes+g.begin;
            g.v=variance[first];g.s=sigma[first];g.r=radius[g.begin];g.component=component[first];
            for(int m=g.begin;m<g.end;++m) {
                size_t j=size_t(k)*modes+m;
                g.common &= variance[j]==g.v && sigma[j]==g.s && radius[m]==g.r && component[j]==g.component;
            }
            groups[size_t(k)*n+i]=g;
        }
    }
};
extern "C" void* direct_context_create(const double* centers,const double* v,const double* s,const double* r,
    const int64_t* c,const double* t,const double* d,int nt,int nn,double spacing,
    const double* w,const int64_t* offsets,const double* ex,int people,int horizon) {
    if(people<=0||horizon<=0||nn<2||nt<=0||spacing<=0)return nullptr;
    for(int i=0;i<people;++i)if(offsets[i+1]<=offsets[i])return nullptr;
    for(int64_t j=0;j<offsets[people]*horizon;++j)if(c[j]<0||c[j]>=nt)return nullptr;
    try { return new Context(centers,v,s,r,c,t,d,nt,nn,spacing,w,offsets,ex,people,horizon); }
    catch(...) {return nullptr;}
}
extern "C" void direct_context_free(void* ptr) {delete static_cast<Context*>(ptr);}
extern "C" int direct_query(void* ptr,const double* positions,int candidates,int threads,double* out) {
    if(!ptr||candidates<0||threads<1)return -1;
    Context& ctx=*static_cast<Context*>(ptr);
    const int h=ctx.horizon,modes=ctx.modes,people=ctx.people;
    ctx.last_cpus.assign(threads,-1);
    // Parallelise independent outputs, NEVER the within-person or within-row sum.
    #pragma omp parallel num_threads(threads) if(threads>1)
    {
    ctx.last_cpus[omp_get_thread_num()]=sched_getcpu();
    #pragma omp for schedule(static)
    for(int64_t row=0;row<int64_t(candidates)*h;++row) {
        int k=row%h; double x=positions[2*row],y=positions[2*row+1],hazard=0.;
        size_t base=size_t(k)*modes;
        for(int i=0;i<people;++i) {
            const Group& g=ctx.groups[size_t(k)*people+i]; double p=0.;
            if(g.common) {
                const double* tab=ctx.table.data()+g.component*ctx.nodes;
                const double* der=ctx.derivative.data()+g.component*ctx.nodes;
                for(int m=g.begin;m<g.end;++m) {
                    double dx=x-ctx.x[base+m],dy=y-ctx.y[base+m];
                    p+=ctx.weights[m]*query_local(std::sqrt(dx*dx+dy*dy),g.v,g.s,g.r,tab,der,ctx.nodes,ctx.spacing);
                }
            } else {
                for(int m=g.begin;m<g.end;++m) {
                    size_t j=base+m;double dx=x-ctx.x[j],dy=y-ctx.y[j];
                    p+=ctx.weights[m]*query_local(std::sqrt(dx*dx+dy*dy),ctx.variance[j],ctx.sigma[j],ctx.radius[m],
                        ctx.table.data()+ctx.component[j]*ctx.nodes,ctx.derivative.data()+ctx.component[j]*ctx.nodes,ctx.nodes,ctx.spacing);
                }
            }
            if(!std::isfinite(p)) {hazard=std::numeric_limits<double>::quiet_NaN();break;}
            hazard-=std::log1p(-std::min(std::max(p*g.existence,0.),1.-1e-12));
        }
        out[row]=hazard;
    }
    }
    return 0;
}
extern "C" int direct_last_cpus(void* ptr,int* out,int capacity) {
    if(!ptr)return 0;
    const auto& cpus=static_cast<Context*>(ptr)->last_cpus;
    if(capacity<int(cpus.size()))return -1;
    std::copy(cpus.begin(),cpus.end(),out);return cpus.size();
}
