#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
#include <cstdlib>
using namespace std;

#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))
const int maxn=1e5+10;
const int MAX=4000;
const ll inf=-0x3f3f3f3f;

ll gcd(ll a, ll b) {return b==0?a:gcd(b,a%b);}
ll lcm(ll a, ll b) {return a/gcd(a,b)*b;}

long long prime[MAX],cnt;
bool isprime[MAX];

void getprime()
{
    cnt=1;
    memset(isprime,1,sizeof(isprime));
    isprime[0]=isprime[1]=0;
    for(long long i=2; i<=MAX; i++)
    {
        if(isprime[i])
            prime[cnt++]=i;
        for(long long j=1; j<cnt&&prime[j]*i<MAX; j++)
        {
            isprime[prime[j]*i]=0;
        }
    }
}

ll s,m;
ll res;

double dp[maxn];
ll ans[maxn];

int main()
{
    cin>>s>>m;
    clr(dp,0);
    getprime();
    for(int i=1;i<=s;i++)
        ans[i]=1;
    for(int i=1;i<=cnt &&prime[i]<=s;i++)
    {
        double tmp=log(double(prime[i]));
        for(int j=s;j>=prime[i];j--)
        {
            for(int k=1,p=prime[i];p<=j;k++)
            {
                if(dp[j-p]+tmp*k>dp[j])
                {
                    dp[j]=dp[j-p]+tmp*k;
                    ans[j]=ans[j-p]*p%m;
                }
                p*=prime[i];
            }
        }
    }
    cout<<ans[s]<<endl;


    return 0;
}
