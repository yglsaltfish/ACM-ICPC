/*
time: 2019/3/1
algorithm: A_star+ Dijkstra
status: no answer
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

typedef pair<int ,int>  pii;

const int maxn=2e3+10;
const double inf=0x3f3f3f;
const int mod=998244353;
const double eps=1e-9;
using namespace std;

priority_queue< pii,vector<pii>,greater<pii> >pq;

struct edge{
    int to,cost;
    edge(int _to,int _cost):to(_to),cost(_cost){}
};
vector <edge> G[maxn];
vector <edge> ZG[maxn];
int dist[maxn];
int n,m,start,ed,k,ans=0;

struct point{
    int v;
    int g,h;
    point(int _v,int _g,int _h):v(_v),g(_g),h(_h){}
    bool operator <(point a) const{
        return h+g > a.h+a.g;
    }
};

void dijk(int s)
{
    for(int i=0;i<=n+1;i++)
        dist[i]=inf;
    dist[s]=0;
    pq.push(pii(0,s));
    while(pq.size())
    {
        pii tp=pq.top();
        pq.pop();
        int v=tp.second;
        if(dist[v]<tp.first)
            continue;
        for(int i=0;i<G[v].size();i++){
            edge e=G[v][i];
            if(dist[e.to]>dist[v]+e.cost){
                dist[e.to]=dist[v]+e.cost;
                pq.push(pii(dist[e.to],e.to));
            }
        }
    }
}

priority_queue <point> que;
int cnt[maxn];
int astar()
{
    clr(cnt,0);
    que.push(point(start,0,dist[start]));
    while(que.size())
    {
        point tp=que.top();
        que.pop();
        int v=tp.v,g=tp.g,h=tp.h;
        cnt[v++];
        if(cnt[v]==k && v==ed)
            return g;
        if(cnt[v]>k)
            continue;
        for(int i=0;i<ZG[v].size();i++)
        {
            edge e=ZG[v][i];
            que.push(point(e.to,e.cost+g,dist[e.to]));
        }
    }
    return -1;
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=0;i<m;i++)
    {
        int x,y,cst;
        scanf("%d%d%d",&x,&y,&cst);
        G[y].push_back(edge(x,cst));
        ZG[x].push_back(edge(y,cst));
    }
    scanf("%d%d%d",&start,&ed,&k);
    if(start==ed)
        k++;
    dijk(ed);
    ans=astar();
    if(ans==-1)
        printf("-1\n");
    else
        printf("%d\n",ans);
    return 0;
}
