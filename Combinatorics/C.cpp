#include <bits/stdc++.h>
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

int t;
int n,k;
int a[maxn],b[maxn];
int dp[maxn];

int main()
{
    cin>>t;
    while(t--)
    {
        cin>>n>>k;
        clr(a,0);clr(b,0);clr(dp,0);
        for(int i=1;i<=k;i++)
            cin>>a[i]>>b[i];
        dp[0]=1;
        for(int i=1;i<=k;i++)
        {
            for(int m=n;m>=a[i];m--)
            {
                for(int j=1;j<=b[i]&&(m>=a[i]*j);j++)
                    dp[m]+=dp[m-a[i]*j];
            }
        }
        cout<<dp[n]<<endl;
    }
    return 0;
}
