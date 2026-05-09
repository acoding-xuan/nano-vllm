# 添加新的模型
1. 阅读config 文件,观察区别. 然后修改 model_runner 文件,使其能够适配多种模型.
2. 在合适的位置添加对应的 model 的文件.
目前支持: llama qwen2 qwen3  qwen3-moe.



## 通用添加模型的策略
1. 读取config 文件. 然后打开 transformers 库, 把transformers 库里面的 相关实现抓取出来,然后和这里的实验进行对应和比较即可...



