# Circular Wiring API

圆形测试治具的自动接线服务：给定按圆周顺序排列的端口与允许连接的无向边（含成本），
选出一份**不交叉的完美匹配**，使总成本最小，并输出可供布线员核验的接线证书。

## 运行

```bash
docker compose up --build
# API: http://localhost:8000  (交互文档 /docs, 健康检查 /healthz)
```

本地开发（Python 3.12）：

```bash
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

## 接口

### `POST /match`

请求体：

```json
{
  "ports": ["a", "b", "c", "d"],
  "edges": [
    {"from": "a", "to": "b", "cost": 1},
    {"from": "c", "to": "d", "cost": 2}
  ]
}
```

- `ports`：2 至 120 个**唯一**的非空 ASCII 端口 id，按圆周顺序给出；奇数个直接 422。
- `edges`：允许连接的无向 pair 与整数成本 `0 ≤ cost ≤ 10^9`。
  重复边（含反向重复）、自连、未知端口均返回 422。

成功响应（`status: "OK"`）：

```json
{
  "status": "OK",
  "pairs": [["a", "b"], ["c", "d"]],
  "total_cost": 3,
  "chords": [
    {
      "pair": ["a", "b"],
      "left_index": 0,
      "right_index": 1,
      "inside_indices": [],
      "inside_port_ids": []
    }
  ]
}
```

- `pairs`：选中的配对。每对端点按下标升序书写，整体排序后取字典序最小的最优解。
- `total_cost`：总成本。
- `chords`：每条弦分隔的下标区间（`left_index`/`right_index` 及被围在弦内侧的下标与端口）。

不存在可行布局时返回 `{"status": "NO_LAYOUT", "detail": ...}`，绝不输出部分配对。

## 算法

区间动态规划，O(n³)（n ≤ 120）。弦 `(l, k)` 把盘面分成 `[l+1, k)` 与 `[k+1, ...)` 两个
独立子问题，因此 `k - l` 必为奇数：

```
dp[l][m] = min over k = l+1, l+3, ..., l+m-1 of
           cost(l, k) + dp[l+1][k-l-1] + dp[k+1][l+m-k-1]
```

每个状态同时保存最优成本与规范配对序列（对内下标升序、整体排序），同成本时按
字典序取最小，保证结果确定唯一。测试用小圆环枚举全部匹配与 DP 对拍，覆盖缺边、
零成本与并列方案。
