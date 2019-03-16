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

typedef pair<int,int>  pii;

const int maxn=2e3+10;
const double inf=0x3f3f3f;
const int mod=998244353;
const double eps=1e-9;
using namespace std;

priority_queue< pii,vector<pii>,greater<pii> >pq;


ll a,b;
ll res;

int main()
{
    int t=0;
    while(scanf("%lld%lld",&a,&b) &&  (a||b))
    {
        printf("Case %d: %lld\n",++t,a*b*(a-1)*(b-1)/4);
    }
    return 0;
}
