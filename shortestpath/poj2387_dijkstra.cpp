/*
time: 2019/2/27
algorithm: Dijkstra+heap
status: accept
*/
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
using namespace std;

#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))

typedef pair<int ,int>  pii;
const int maxn=2e5+10;
const int inf=0x3f3f3f3f;
const int mod=998244353;

ll gcd(ll a, ll b){return b==0?a:gcd(b,a%b);}

priority_queue<pii,vector<pii>,greater<pii> >pq;
struct edge{
    int to,cost;
};
vector <edge> G[maxn];
int dist[maxn];
int t,n;
void dijk(int s)
{
    for(int i=1;i<=n;i++)
        dist[i]=inf;
    dist[s]=0;
    pq.push(make_pair(0,s));
    while(!pq.empty())
    {
        pii u=pq.top();
        pq.pop();
        int x=u.second;
        for(int i=0;i<G[x].size();i++)
        {
            edge e=G[x][i];
            if(dist[e.to]>dist[x]+e.cost)
            {
                dist[e.to]=dist[x]+e.cost;
                pq.push(make_pair(dist[e.to],e.to));
            }
        }
    }
}

int main()
{
    scanf("%d%d",&t,&n);
    for(int i=0;i<t;i++)
    {
        int From,To,Cost;
        scanf("%d%d%d",&From,&To,&Cost);
        edge tmp;
        tmp.to=To;tmp.cost=Cost;
        G[From].push_back(tmp);
        tmp.to=From;
        G[To].push_back(tmp);
    }
    dijk(n);
    printf("%d\n",dist[1]);
    return 0;
}
