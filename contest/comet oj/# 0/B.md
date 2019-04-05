## 旅途

---

### 题目描述：

小象来到了百特国旅游。百特国有 *n* 所城市，这些城市围成了一个圈。小象按照顺时针给这些城市从 1 到 *n* 编号，其中 11 号城市是他最先到达的城市。

小象共有 *m* 天的旅游时间。在第一天里，小象会选择游玩 1 号城市。而在接下来的每一天，他有 *p*% 的概率去前一天游玩城市顺时针方向的下一个城市游玩，有 *q*% 的概率去逆时针方向的下一个城市游玩，也有 (100 - *p* - *q*)% 的概率继续停留在前一天的城市游玩。 

他很好奇他平均可以游玩到多少个不同的城市，你可以帮他计算一下吗？不妨设他能恰好游玩到 i*i* 个城市的概率是 f(i)*f*(*i*)，在给定正整数 k*k* 的情况下，请你计算 $100^{m - 1} \sum\limits_{i = 1}^{n}{i^k f(i)}对 (10^9 + 7)$ 取模的值。

题目链接：[<https://www.cometoj.com/contest/34/problem/B?problem_id=1474>]

---

### 解题思路：

小象游玩的城市可以看成一个区间，我们只关心这个去间的长度，和小象在这个区间的位置。

于是可以令 *dp(i , j , k )* 表示小象在第 *i* 天游玩的城市，顺时针方向已经有 *j* 个城市游玩，逆时针方向有 *k* 个城市已经游玩 的 概率之和。所以状态转移的时候只需要枚举 *i+1* 和 *i*  之间城市的相对位置。

然后 $i^{k}$ 使用快速幂算，最后计算的时候，需 *k=min(j+i+1,n)* ，因为 *i+k+1<n* 。

因为 *p , q* 给的都是 百分数，所以不需要计算 $100^{m-1}$ 。

以下是代码:

``` c++
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
#define db double

const int mod = 1e9 + 7;
const int maxn = 5e2 + 10;

ll gcd(ll a, ll b) { return b == 0 ? a : gcd(b, a % b); }
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
ll qpow(ll a,ll b)
{
    ll res = 1;
    while (b)
    {
        if (b & 1)
            res = res * a % mod;
        b >>= 1;
        a = a * a % mod;
    }
    return res % mod;
}

ll dp[maxn][maxn][maxn];

int t;
int n, m, k, p, q;
ll res;

int main()
{
    cin >> t;
    while(t--)
    {

        cin >> n >> m >> k >> p >> q;

        for (int i = 0; i <= m;i++)
            for (int j = 0; j <= m;j++)
                for (int k = 0; k <= m;k++)
                    dp[i][j][k] = 0;

        dp[1][0][0] = 1;

        for (int i = 1; i <= m; i++)
        {
            for (int j = 0; j < i; j++)
            {
                for (int k = 0; j+k < i; k++)
                {

                    dp[i + 1][max(0,j-1)][k + 1] += (dp[i][j][k] * p % mod);
                    dp[i + 1][max(0,j-1)][k + 1] %= mod;

                    dp[i + 1][j + 1][max(k-1,0)] += (dp[i][j][k] *q % mod);
                    dp[i + 1][j + 1][max(k-1,0)] %= mod;

                    dp[i + 1][j][k] += (dp[i][j][k] * (100 - p - q) % mod);
                    dp[i + 1][j][k] %= mod;
                }
            }
        }

        res = 0;
        for (int i = 0; i <= m; i++)
        {
            for (int j = 0; j <= m;j++)
            {
                    res = (res + (1ll)*qpow(min(i + j + 1, n), k) * dp[m][i][j] % mod) % mod;
            }
        }
        cout << res%mod << endl;
    }
    return 0;
}



```