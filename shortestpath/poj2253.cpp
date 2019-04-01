/*
time: 2019/2/28
algorithm: Dijkstra+heap
status: accept
*/
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
using namespace std;

#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))

typedef pair<double ,int>  pii;

const int maxn=2e3+10;
const double inf=0x3f3f3f3f;
const int mod=998244353;
const double eps=1e-9;

ll gcd(ll a, ll b){return b==0?a:gcd(b,a%b);}

double getdist(double x1,double y1,double x2,double y2){return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));}

priority_queue <pii,vector<pii>,greater<pii> > pq;

int n;
struct edge{
    int to;
    double cost;
};

vector <edge> G[maxn];
double dist[maxn];
double ans;

void dijk(int s)
{
    ans=0;
    for(int i=0;i<=n;i++)
        dist[i]=inf;
    dist[s]=0;
    pq.push(pii(0.0,s));
    while(!pq.empty())
    {
        pii u=pq.top();
        pq.pop();
        int x=u.second;
        if(dist[x]>u.first) continue;

        if(dist[x]!=inf)    ans=max(dist[x],ans);

        if(x==2)    return ;

        for(int i=0;i<G[x].size();i++)
        {
            edge e=G[x][i];
            if(dist[e.to]>e.cost)
            {
                dist[e.to]=e.cost;
                pq.push(pii(dist[e.to],e.to));
            }
        }
    }
}

double x[maxn],y[maxn];
int main()
{
    int t=0;
    while(~scanf("%d",&n) && n)
    {
        clr(x,0);
        clr(y,0);
        while(!pq.empty())
            pq.pop();
        for(int i=0;i<=n;i++)
            G[i].clear();

        for(int i=1;i<n;i++)
            scanf("%lf%lf",&x[i],&y[i]);
        edge tmp;
        for(int i=1;i<=n;i++)
        {
            for(int j=1;j<i;j++)
            {
                if(i==j)
                    continue;
                tmp.to=j;
                tmp.cost=getdist(x[i],y[i],x[j],y[j]);
                G[i].push_back(tmp);
                tmp.to=i;
                G[j].push_back(tmp);
            }
        }
        if(n==2)
        {
            printf("Scenario #%d\n",++t);
            printf("Frog Distance = %.3f\n\n",getdist(x[1],y[1],x[2],y[2]));
            continue;
        }
        dijk(1);
        printf("Scenario #%d\n",++t);
        printf("Frog Distance = %.3f\n\n",ans);
    }
    return 0;
}
