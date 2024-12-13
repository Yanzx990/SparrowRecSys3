# 二次开发

## 数据扩容

![](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241212160619506.png?raw=true)

![image-20241212161340623](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241212161340623.png?raw=true)



### 逻辑：

在主函数中调用datamanager，把数据装载进去。

从代码中可以看出要扩容数据，必须包含

1. movies.csv中的 movieId,title,genres![image-20241212183341503](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241212183341503.png?raw=true)
2. item2vecembedding
3. links
4. ratings



## 前端修改

### 加入搜索功能：

在index.html中找到搜索栏相关的源代码，通过设置button来触发performsearch函数

![image-20241213194248752](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213194248752.png?raw=true)

![image-20241213195426901](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213195426901.png?raw=true)

在recsys.js中写出searchMovie这个function

![image-20241213201211030](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213201211030.png?raw=true)

在RecSysServer中定义好前端index.html中发起的ajax请求"searchmovie"所绑定的服务

![image-20241213201553848](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213201553848.png?raw=true)

重新写一个类"SearchMovieSerivce",主要的功能是通过datamanager类来寻找匹配的电影，将其返回成一个list，最后再转为json文件，重新发回前端用于动态渲染.

![image-20241213201652978](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213201652978.png?raw=true)

在datamanager里新写一个方法，来匹配数据中的电影名称

![image-20241213202608783](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213202608783.png?raw=true)

又加了一个按下回车键可以开始搜索：

![](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213203245593.png?raw=true)

![](https://github.com/Yanzx990/SparrowRecSys3/blob/yanzx/docs/image-20241213203258243.png?raw=true)

## 网页汉化

## 模型重新训练

## 





