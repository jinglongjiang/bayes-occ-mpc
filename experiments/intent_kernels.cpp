// Experimental adapter; original RVO2 sources and LP implementations stay unchanged.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct RiskNode { double x,y,r,w; int left,right,mode; };
struct RiskTree {
    std::vector<RiskNode> nodes;
    std::vector<int> roots;
    const double* centers; const double* weights; int horizon;
    int build(std::vector<int>& ids,int begin,int end,int k) {
        int node=nodes.size(); nodes.push_back({0,0,0,0,-1,-1,ids[begin]});
        RiskNode n=nodes[node]; double low[2]={1e300,1e300},high[2]={-1e300,-1e300};
        for(int z=begin;z<end;++z) {
            int m=ids[z]; double w=weights[m];n.w+=w;
            for(int d=0;d<2;++d) {double v=centers[2*(m*horizon+k)+d];low[d]=std::min(low[d],v);high[d]=std::max(high[d],v);}
        }
        n.x=(low[0]+high[0])*.5;n.y=(low[1]+high[1])*.5;
        for(int z=begin;z<end;++z) {
            int j=2*(ids[z]*horizon+k); double dx=centers[j]-n.x,dy=centers[j+1]-n.y;
            n.r=std::max(n.r,std::sqrt(dx*dx+dy*dy));
        }
        if(end-begin>1 && n.r>0.) {
            int axis=high[1]-low[1]>high[0]-low[0],mid=(begin+end)/2;
            std::nth_element(ids.begin()+begin,ids.begin()+mid,ids.begin()+end,[&](int a,int b){
                double x=centers[2*(a*horizon+k)+axis],y=centers[2*(b*horizon+k)+axis];return x==y?a<b:x<y;
            });
            n.left=build(ids,begin,mid,k);n.right=build(ids,mid,end,k);
        }
        nodes[node]=n;return node;
    }
};
#include "Agent.h"
#include "KdTree.h"
#include "RVOSimulator.h"

namespace RVO {
class BatchContext {
public:
    RVOSimulator sim;
    BatchContext(const double* state, const double* others, int count)
      : sim(.25f,10.f,10,5.f,5.f,.3f,1.f) {
        sim.addAgent(Vector2(state[0],state[1]),10.f,10,5.f,5.f,
                     float(state[4]+.01),1.f,Vector2(state[2],state[3]));
        for(int i=0;i<count;++i) {
            const double* o=others+5*i;
            sim.addAgent(Vector2(o[0],o[1]),10.f,10,5.f,5.f,
                         float(o[4]+.01),1.f,Vector2(o[2],o[3]));
        }
        sim.setAgentPrefVelocity(0,Vector2(0.f,0.f));
        sim.kdTree_->buildAgentTree();
        sim.agents_[0]->computeNeighbors();
        // Also performs one unused target LP; never solves or advances neighbors.
        sim.agents_[0]->computeNewVelocity();
    }
    void solve(const double* prefs, int count, double* out) const {
        const Agent* agent=sim.agents_[0];
        for(int i=0;i<count;++i) {
            Vector2 v;
            const Vector2 p(prefs[2*i],prefs[2*i+1]);
            size_t failure=linearProgram2(agent->orcaLines_,agent->maxSpeed_,p,false,v);
            if(failure<agent->orcaLines_.size())
                linearProgram3(agent->orcaLines_,0,failure,agent->maxSpeed_,v);
            out[2*i]=v.x();out[2*i+1]=v.y();
        }
    }
};
}

extern "C" {
void* context_create(const double* state,const double* neighbors,int n) {
    return new RVO::BatchContext(state,neighbors,n);
}
void context_free(void* p) {delete static_cast<RVO::BatchContext*>(p);}
double ttc_scale(const double* state,const double* neighbors,int n) {
    double closest=-1.;
    for(int i=0;i<n;++i) {
        const double* o=neighbors+5*i;
        double px=o[0]-state[0],py=o[1]-state[1],vx=o[2]-state[2],vy=o[3]-state[3];
        double a=vx*vx+vy*vy;
        if(a<1e-8)continue;
        double radius=state[4]+o[4],b=2.*(px*vx+py*vy),c=px*px+py*py-radius*radius;
        double disc=b*b-4.*a*c;
        if(disc<=0.)continue;
        double root=std::sqrt(disc),t1=(-b-root)/(2.*a),t2=(-b+root)/(2.*a);
        if(t2<0.)continue;
        double t=t1>0.?t1:t2;
        if(closest<0. || t<closest)closest=t;
    }
    return closest<0. || closest>=2.?1.:std::max(0.,std::min(1.,closest/2.));
}

void behavior_forward(void* shared,const double* initial,const double* neighbors,int others,
                      const double* goals,int modes,int horizon,double initial_ratio,double* output) {
    std::vector<double> predicted(size_t(horizon)*others*5);
    for(int k=0;k<horizon;++k) for(int i=0;i<others;++i) {
        const double* o=neighbors+5*i;double* q=predicted.data()+(k*others+i)*5;
        q[0]=o[0]+(k*.25)*o[2];q[1]=o[1]+(k*.25)*o[3];
        q[2]=o[2];q[3]=o[3];q[4]=o[4];
    }
    for(int i=0;i<modes;++i) {
        double state[5];std::copy(initial,initial+5,state);
        double memory[2]={state[2],state[3]};
        for(int k=0;k<horizon;++k) {
            const double* people=others?predicted.data()+k*others*5:neighbors;
            double pref[2]={goals[2*i]-state[0],goals[2*i+1]-state[1]};
            double speed=std::sqrt(pref[0]*pref[0]+pref[1]*pref[1]);
            if(speed>1.){pref[0]/=speed;pref[1]/=speed;}
            if(others) {
                double scale=k==0?initial_ratio:ttc_scale(state,people,others);
                pref[0]=.5*(pref[0]*scale)+.5*memory[0];pref[1]=.5*(pref[1]*scale)+.5*memory[1];
                memory[0]=pref[0];memory[1]=pref[1];
            }
            double desired[2];
            if(k==0)static_cast<RVO::BatchContext*>(shared)->solve(pref,1,desired);
            else {RVO::BatchContext context(state,people,others);context.solve(pref,1,desired);}
            double vn=std::sqrt(state[2]*state[2]+state[3]*state[3]);
            double dn=std::sqrt(desired[0]*desired[0]+desired[1]*desired[1]);
            if(vn>=.1 && dn>=.1) {
                double angle=std::atan2(state[2]*desired[1]-state[3]*desired[0],state[2]*desired[0]+state[3]*desired[1]);
                angle=std::max(-.375,std::min(.375,angle));double c=std::cos(angle),s=std::sin(angle);
                double ux=state[2]/vn,uy=state[3]/vn;
                desired[0]=(dn*c)*ux+(dn*(-s))*uy;
                desired[1]=(dn*s)*ux+(dn*c)*uy;
            }
            double dx=desired[0]-state[2],dy=desired[1]-state[3];
            double factor=std::min(1.,.125/std::max(std::sqrt(dx*dx+dy*dy),1e-12));
            state[2]+=dx*factor;state[3]+=dy*factor;
            state[0]+=.25*state[2];state[1]+=.25*state[3];
            output[2*(i*horizon+k)]=state[0];output[2*(i*horizon+k)+1]=state[1];
        }
    }
}

void fused_bounds(const double* positions,int candidates,int horizon,
    const double* centers,int modes,const double* variance,const double* sigma,
    const double* radius,const double* a,const int64_t* component,
    const double* table,const double* derivative,int nodes,double spacing,double error,
    const double* weights,const int64_t* offsets,const double* existence,int people,
    double farprob,double inside,double* lower,double* upper) {
    for(int n=0;n<candidates;++n) for(int k=0;k<horizon;++k) {
        double hl=0.,hu=0.;
        for(int i=0;i<people;++i) {
            double pl=0.,pu=0.;
            for(int64_t m=offsets[i];m<offsets[i+1];++m) {
                const int64_t j=m*horizon+k;
                const double dx=positions[2*(n*horizon+k)]-centers[2*j];
                const double dy=positions[2*(n*horizon+k)+1]-centers[2*j+1];
                const double d=std::sqrt(dx*dx+dy*dy), r=radius[m];
                const double z=(d-r)/sigma[j];
                double lo,hi;
                if(variance[j]<1e-9) lo=hi=double(d<=r);
                else if(z>=8.) lo=hi=farprob;
                else if(z < -8.) {lo=inside;hi=1.;}
                else {
                    const double index=(z+8.)/spacing;
                    const int left=std::min(std::max(int(std::floor(index)),0),nodes-2);
                    const double t=std::min(std::max(index-left,0.),1.),t2=t*t,t3=t2*t;
                    const int64_t slot=component[j]*nodes+left;
                    const double ql=table[slot],qr=table[slot+1];
                    const double approx=((2*t3-3*t2+1)*ql+(t3-2*t2+t)*spacing*derivative[slot]
                        +(-2*t3+3*t2)*qr+(t3-t2)*spacing*derivative[slot+1]);
                    lo=approx-error;hi=approx+error;
                    if(a[j]+(-8.+left*spacing)>=0.) {lo=std::max(lo,qr);hi=std::min(hi,ql);}
                    lo=std::max(0.,lo-1e-12);hi=std::min(1.,hi+1e-12);
                }
                pl+=weights[m]*lo;pu+=weights[m]*hi;
            }
            hl-=std::log1p(-std::min(std::max(pl*existence[i],0.),1.-1e-12));
            hu-=std::log1p(-std::min(std::max(pu*existence[i],0.),1.-1e-12));
        }
        lower[n*horizon+k]=hl;upper[n*horizon+k]=hu;
    }
}

int exact_prepare(const double* positions,int candidates,int horizon,const double* centers,
    int modes,const double* variance,const double* radius,double farprob,
    double* probability,double* scaled,double* noncentral,int64_t* slots) {
    int used=0;
    for(int n=0;n<candidates;++n) for(int m=0;m<modes;++m) for(int k=0;k<horizon;++k) {
        int j=m*horizon+k, slot=(n*modes+m)*horizon+k;
        double dx=positions[2*(n*horizon+k)]-centers[2*j];
        double dy=positions[2*(n*horizon+k)+1]-centers[2*j+1];
        double d=std::sqrt(dx*dx+dy*dy),v=variance[j],r=radius[m];
        if(v<1e-9) probability[slot]=double(d<=r);
        else if(d-r>=8.*std::sqrt(v)) probability[slot]=farprob;
        else {scaled[used]=r*r/std::max(v,1e-9);noncentral[used]=d*d/std::max(v,1e-9);
              slots[used++]=slot;}
    }
    return used;
}
void aggregate_exact(const double* probability,int candidates,int horizon,int modes,
    const double* weights,const int64_t* offsets,const double* existence,int people,double* out) {
    for(int n=0;n<candidates;++n) for(int k=0;k<horizon;++k) {
        double h=0.;
        for(int i=0;i<people;++i) {
            double p=0.;
            for(int64_t m=offsets[i];m<offsets[i+1];++m)
                p+=weights[m]*probability[(n*modes+m)*horizon+k];
            h-=std::log1p(-std::min(std::max(p*existence[i],0.),1.-1e-12));
        }
        out[n*horizon+k]=h;
    }
}

void* risk_tree_build(const double* centers,const double* weights,const int64_t* offsets,int people,int horizon) {
    RiskTree* tree=new RiskTree();tree->centers=centers;tree->weights=weights;tree->horizon=horizon;
    for(int i=0;i<people;++i) for(int k=0;k<horizon;++k) {
        std::vector<int> ids;for(int m=offsets[i];m<offsets[i+1];++m)ids.push_back(m);
        tree->roots.push_back(tree->build(ids,0,ids.size(),k));
    }
    return tree;
}
void risk_tree_free(void* tree) {delete static_cast<RiskTree*>(tree);}

double point_query(double d,int j,int m,const double* variance,const double* sigma,const double* radius,
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

// mode 0: clean full-mode interpolation; mode 1: adaptive per-person tree bounds.
void approximate_risk(int mode,void* pointer,const double* positions,int candidates,int horizon,
    const double* centers,const double* variance,const double* sigma,const double* radius,
    const int64_t* component,const double* table,const double* derivative,int nodes,double spacing,
    double error,const double* weights,const int64_t* offsets,const double* existence,int people,
    double epsilon,double* lower,double* upper,int64_t* counts) {
    RiskTree* tree=static_cast<RiskTree*>(pointer);std::vector<int> stack;
    counts[0]=counts[1]=counts[2]=0;
    for(int n=0;n<candidates;++n) for(int k=0;k<horizon;++k) {
        double hl=0.,hu=0.,x=positions[2*(n*horizon+k)],y=positions[2*(n*horizon+k)+1];
        for(int i=0;i<people;++i) {
            double pl=0.,pu=0.;
            if(mode==0) {
                for(int m=offsets[i];m<offsets[i+1];++m) {
                    int j=m*horizon+k;double dx=x-centers[2*j],dy=y-centers[2*j+1];
                    pl+=weights[m]*point_query(std::sqrt(dx*dx+dy*dy),j,m,variance,sigma,radius,component,table,derivative,nodes,spacing);
                    ++counts[0];
                }
                pu=pl;
            } else {
                stack.clear();stack.push_back(tree->roots[i*horizon+k]);
                while(!stack.empty()) {
                    RiskNode a=tree->nodes[stack.back()];stack.pop_back();++counts[1];
                    if(a.w==0.)continue;
                    int m=a.mode,j=m*horizon+k;double dx=x-a.x,dy=y-a.y,d=std::sqrt(dx*dx+dy*dy);
                    double lo=point_query(d+a.r,j,m,variance,sigma,radius,component,table,derivative,nodes,spacing);
                    double hi=point_query(std::max(0.,d-a.r),j,m,variance,sigma,radius,component,table,derivative,nodes,spacing);
                    if(!std::isfinite(lo) || !std::isfinite(hi)) {
                        lower[0]=upper[0]=std::numeric_limits<double>::quiet_NaN();return;
                    }
                    double pad=variance[j]<1e-9?0.:error+1e-12;
                    lo=std::max(0.,lo-pad);hi=std::min(1.,hi+pad);counts[0]+=2;
                    if(a.left<0 || hi-lo<=epsilon) {pl+=a.w*lo;pu+=a.w*hi;++counts[2];}
                    else {stack.push_back(a.right);stack.push_back(a.left);}
                }
            }
            if(!std::isfinite(pl) || !std::isfinite(pu)) {
                lower[0]=upper[0]=std::numeric_limits<double>::quiet_NaN();return;
            }
            hl-=std::log1p(-std::min(std::max(pl*existence[i],0.),1.-1e-12));
            hu-=std::log1p(-std::min(std::max(pu*existence[i],0.),1.-1e-12));
        }
        lower[n*horizon+k]=hl;upper[n*horizon+k]=hu;
    }
}
}
