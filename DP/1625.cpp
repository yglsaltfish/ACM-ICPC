/*
UVa 1625  �����Ŀ�������ù�����������Ż���
dp[i][j]   i,j�ֱ��ʾ��һ���ַ���ȡ����i�����ڶ����ַ���ȡ����j������СLֵ
*/

#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;

#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))

const int maxn=5010;
const int inf=0x3f3f;
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}

int t;
int dp[maxn][maxn],c[maxn][maxn];
int sa[26],ea[26],sb[26],eb[26];
int la,lb,v1,v2;
char a[maxn],b[maxn];

int main()
{
    scanf("%d",&t);
    while(t--)
    {
        scanf("%s%s",a+1,b+1);
        la=strlen(a+1);
        lb=strlen(b+1);
        for(int i=1; i<=la; i++)
            a[i]-='A';
        for(int i=1; i<=lb; i++)
            b[i]-='A';
        clr(sa,inf);
        clr(sb,inf);
        clr(ea,-inf);
        clr(eb,-inf);

        for(int i=1; i<=la; i++)
        {
            sa[a[i]]=min(sa[a[i]],i);
            ea[a[i]]=i;
        }
        for(int i=1; i<=lb; i++)
        {
            sb[b[i]]=min(sb[b[i]],i);
            eb[b[i]]=i;
        }
        dp[0][0]=c[0][0]=0;
        for(int i=0; i<=la; i++)
        {
            for(int j=0; j<=lb; j++)
            {
                if(!i && ! j)
                    continue;
                v1=v2=inf;
                if(i) v1=dp[i-1][j]+c[i-1][j];
                if(j) v2=dp[i][j-1]+c[i][j-1];
                dp[i][j]=min(v1,v2);
                if(i)
                {
                    c[i][j]=c[i-1][j];
                    if(sa[a[i]]==i && sb[a[i]]>j) c[i][j]++;
                    if(ea[a[i]]==i && eb[a[i]]<=j) c[i][j]--;
                }
                else if(j)
                {
                    c[i][j] = c[i][j-1];
                    if(sb[b[j]] == j &&sa[b[j]]>i) c[i][j]++;
                    if(eb[b[j]] == j &&ea[b[j]]<=i)c[i][j]--;
                }
            }
        }
        printf("%d\n",dp[la][lb]);
       // cout<<0x3f<<endl;
    }
    return 0;
}
