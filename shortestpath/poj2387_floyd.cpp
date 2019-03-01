/***********************************************
 * Author: fisty
 * Created Time: 2015/2/1 20:14:57
 * File Name   : poj2449.cpp
 *********************************************** */
#include <iostream>
#include <cstring>
#include <cmath>
#include <queue>
#include <vector>
#include <cstdio>
#include <algorithm>
using namespace std;
#define Debug(x) cout << #x << " " << x <<endl
#define MAX_N 1100
const int INF = 0x3f3f3f3f;
typedef long long LL;
typedef pair<int, int> P;
int n, m;
int start, end, K;
//���·
struct edge{
    int to;
    int cost;
    edge(int _to, int _cost):to(_to), cost(_cost){}
};
vector<edge> _G[MAX_N];
int dist[MAX_N];                     //�յ㵽��������·

void dijsktra(int t){
    //t -> i
    priority_queue <P, vector<P>, greater<P> > que;
    memset(dist, 0x3f, sizeof(dist));
    dist[t] = 0;
    que.push(P(dist[t], t));
    while(que.size()){
        P q = que.top(); que.pop();
        int v = q.second;
        if(dist[v] < q.first) continue;
        for(int i = 0;i < _G[v].size(); i++){
            edge e = _G[v][i];
            if(dist[e.to] > dist[v] + e.cost){
                dist[e.to] = dist[v] + e.cost;
                que.push(P(dist[e.to], e.to));
            }
        }
    }
}

//Astar
struct node{
    //���������ȶ��еĽ��
    //f = g + h, fС�����ȳ���
    int v;
    int g;               //gΪ�ڵ�start��i��ʵ��ֵ
    int h;               //hΪ�ڵ�i��end�����ٹ���ֵ
    node(int _v,int _g, int _h):v(_v), g(_g), h(_h){}
    bool operator < (node a) const{
        return h + g > a.h + a.g;
    }
};
int cnt[MAX_N];                 //�ڵ�ļ�����������K���֦����Ϊ�ǵ�K�̣�1�����㲻�ᳬ��K��
vector<edge> G[MAX_N];
int A_star(){
    priority_queue <node> que;
    memset(cnt, 0, sizeof(cnt));
    que.push(node(start, 0, dist[start]));          //��ӵ���ϢΪ ���,��������ֵ�� hֵ
    while(que.size()){
        node e = que.top(); que.pop();
        int v = e.v, g = e.g, h = e.h;
        cnt[v]++;
        if(cnt[v] == K && v == end) return g ;
        if(cnt[v] > K) continue;
        for(int i = 0;i < G[v].size(); i++){
            edge e = G[v][i];
            que.push(node(e.to, e.cost + g, dist[e.to]));
        }
    }
    return -1;
}
int main(){
    //freopen("in.txt", "r", stdin);
    cin.tie(0);
    ios::sync_with_stdio(false);
    cin >> n >> m;
    for(int i = 0;i < m; i++){
        int u, v, cost;
        cin >> u >> v >> cost;
        G[u].push_back(edge(v, cost));
        _G[v].push_back(edge(u, cost));
   }
    cin >> start >> end >> K;
    if(start == end){  //����յ���ͬ,���·Ϊ0
        ++K;
    }
    dijsktra(end);
    int ans = A_star();
    if(ans == -1)
        cout << -1 << endl;
    else
        cout << ans << endl;
    return 0;
}
