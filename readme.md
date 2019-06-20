# 实时人脸贴纸：基于3D人脸模型

作者：
唐雯豪(1800013088), 韩宇栋(1800013097), 崔轩宁(1800013083)

## 代码说明

这是一个经过精简的项目文件，内容比我们的开发文件少很多很多。

`main.py` 是主程序，运行后可以进入图形化界面。

默认使用cpu，如果要改用GPU，请在代码中将`a_fb`换成`a_mbn`并将`DEVICE`改为`GPU`。

`a_fb.py`和`a_mbn.py`是主要处理代码，不同之处在于一个使用faceboxes另一个使用mobilenet。

`MEOW3DDFA`里是3DDFA部分，其中`a_3ddfa.py`是接口。

`deepface`是使用GPU的人脸检测，方法是mobilenet

`FaceBoxes`是使用CPU的人脸检测，方法是faceboxes

(PS:我们在后期版本中弃用了MTCNN)

`filters.py`是添加贴纸的代码。

`big-final.ui`和`ui.py`等是GUI相关文件。

如果有任何运行问题，请联系 tangwh@pku.edu.cn