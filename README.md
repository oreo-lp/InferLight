## InferLight

Tools for batch inference.

### Test result
#### Bert inference
- Single text inference.

```
Concurrency Level:      32
Time taken for tests:   10.164 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      202000 bytes
HTML transferred:       111000 bytes
Requests per second:    98.39 [#/sec] (mean)
Time per request:       325.234 [ms] (mean)
Time per request:       10.164 [ms] (mean, across all concurrent requests)
Transfer rate:          19.41 [Kbytes/sec] received
```

- Batch inference
```
Concurrency Level:      32
Time taken for tests:   4.019 seconds
Complete requests:      1000
Failed requests:        999
   (Connect: 0, Receive: 0, Length: 999, Exceptions: 0)
Total transferred:      202978 bytes
HTML transferred:       111978 bytes
Requests per second:    248.79 [#/sec] (mean)
Time per request:       128.620 [ms] (mean)
Time per request:       4.019 [ms] (mean, across all concurrent requests)
Transfer rate:          49.32 [Kbytes/sec] received
# 上面的Failed requests主要问题的Length导致的，这个是因为网页都动态的，返回的response大小不一致，所以是Failed。  
这个可以忽略
```



| 模型 | 显存 | GPU利用率 | 内存 | CPU利用率 | QPS |  
| :--- | :---: | :---: | :---: |  :---: | :---: |  
| flask-bert | 2349M | 28% | 2.9G | 218% |  45 |    
| fastapi-bert | 1395M | 40% | 2.6G | 100% | 96 |  
| inferlight-bs1 | 2711M | 40% | 5.2G | 121% |  101  | 
| inferlight-bs2 | 2711M | 40% | 5.2G | 137% | 207  | 
| inferlight-bs4 | 2711M | 40% | 5.2G | 152% | 396  |   
| inferlight-bs8 | 2711M | 40% | 5.2G | 180% | 659  |  
| inferlight-bs16 | 2711M | 40% | 5.2G | 220% | 756  |  
