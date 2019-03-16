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

const int MAX=27;
short s[MAX];
int sub[100];

void mult(int *sub,int sum,int m)
{
    int Base=10000,p=0;
    for(int i=1; i<=sub[0]; ++i)
    {
        sub[i]=sub[i]*sum+p;
        p=sub[i]/Base;
        sub[i]=sub[i]%Base;
    }
    while(p)
    {
        sub[0]++;
        sub[sub[0]]=p%Base;
        p=p/Base;
    }
    for(int i=sub[0]; i>=1; --i)
    {
        sub[i]+=p;
        p=(sub[i]%m)*Base;
        sub[i]=sub[i]/m;
    }
    while(!sub[sub[0]])sub[0]--;
}

void cbm(int n,int m)
{
    for(int i=1; i<=m; ++i)
        mult(sub,n-m+i,i);
    return;
}

int main()
{
    int n;
    while(cin>>n,n)
    {
        int sum=0;
        sub[0]=sub[1]=1;
        for(int i=0; i<n; ++i)
        {
            cin>>s[i];
            sum+=s[i];
        }
        for(int i=0; i<n-1; ++i)
        {
            cbm(sum,s[i]);
            sum-=s[i];
        }
        cout<<sub[sub[0]];
        for(int i=sub[0]-1; i>=1; --i)
        {
            cout<<setfill('0')<<setw(4)<<sub[i];
        }
        cout<<endl;
    }
    return 0;
}
