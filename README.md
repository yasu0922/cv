# 画像を認識するサンプル

Python+OpenCVで対象画像内に検索画像が存在する確認して該当箇所を短形で囲むスクリプトサンプル。

# DEMO
```bash
python Confirm_with_image.py query.jpg train.jpg, checkpoint.jpg
```

* __query.jpg：検索画像__
![query](https://user-images.githubusercontent.com/29660278/100187053-9af7e000-2f2a-11eb-88e6-6ddc0bbb8e1e.jpg)
* __train.jpg：検索対象が含まれる画像__
![train](https://user-images.githubusercontent.com/29660278/100186982-77349a00-2f2a-11eb-9f66-3d772139afda.jpg)
* __checkpoint.jpg：最終的に確認したい箇所の画像__

![checkpoint](https://user-images.githubusercontent.com/29660278/100187094-ac40ec80-2f2a-11eb-9487-0ab915df1e8c.jpg)

# Instllation
## python
python 3.7.7(3.6や3.8でも動作すると思います)

## packages
Anacondaの使用をお勧めします。

* numpy 1.18.5
* opencv 4.2.0(3.x系とシグネチャが異なる関数があるかも)

# Note
内部で射影変換を行っているのである程度の奥行きにも対応しますが極力平面で撮影した方が良いです。
