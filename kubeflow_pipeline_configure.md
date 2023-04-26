# kubeflow pipeline 구성 가이드
참고 : https://medium.com/kubwa/part-1-image-classification-on-mlops-424625c4027c
## 1. 클라이언트 (local 또는 별도 kubernetes 환경) 에서</BR> 각 Pipeline의 컨테이너 생성
## docker 컨테이너 생성 및 docker hub 업로드

### 구성 파일 다운로드
```commandline
git clone https://github.com/white091612/kubeflow.git
```
### 다운로드 된 파일 구성도
```commandline
kubernets
└ hyperparameter
  └ Dockerfile
    hyperparameter-wandb.py
    util.py
└ preprocess
  └ Dockerfile
    preprocess.py
└ train
  └ Dockerfile
    train.py
    util.py
└ test
  └ Dockerfile
    test.py
    util.py
```
### 각 폴더에서 컨테이너 생성
```
cd ./kubernets/preprocess
docker build -t <docker account>/<container name>:<tag> .
docker push <docker account>/<container name>:<tag>

cd ..
cd hyperparameter
docker build -t <docker account>/<container name>:<tag> .
docker push <docker account>/<container name>:<tag>

cd ..
cd train
docker build -t <docker account>/<container name>:<tag> .
docker push <docker account>/<container name>:<tag>

cd ..
cd test
docker build -t <docker account>/<container name>:<tag> .
docker push <docker account>/<container name>:<tag>
```
### pipeline 생성을 위한 .yaml 파일 생성
pipeline.py는 위에서 생성된 도커 이미지 정보가 들어있어, 위에 입력한 정보에 맞게 수정해야하며</BR>
.yaml 파일의 이름도 수정해야 한다.
```commandline
cd ./kubernetes
python pipeline.py
```
### kubeflow Dashboard 접속 후 Upload pipeline 클릭
```
kubectl port-forward --address 0.0.0.0 svc/istio-ingressgateway -n istio-system 8080:80
```
### 
<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/234530346-d6b37981-18c2-426d-8e58-223c121f0961.png" title="central-dashboard"/>
</p>

### 생성된 .yaml 파일을 업로드 후 Create 클릭하여 파이프라인 생성
<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/234531656-bee68635-1bba-4202-a7ac-bf348a6682e3.png" title="central-dashboard"/>
</p>

### Create Run
<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/234531914-f2caa37d-d52f-4bb3-93db-b37adb0a698b.png" title="central-dashboard"/>
</p>

### Experiment 선택(없으면 왼쪽 메뉴에서 생성)
<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/234532001-5faf1324-b500-4fce-ab6f-7e95833de07d.png" title="central-dashboard"/>
</p>

### Run parameters 입력
<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/234532291-da32c68b-4f59-43c7-83cd-39a95d53f575.png" title="central-dashboard"/>
</p>

### preprocess
```commandline
model_hyp_train_test : ""
preprocess : "yes"
model_path : ""
device : "cpu", "0", "1", "2", "3" 중 입력
```
위 처럼 입력 시 폴더 구분 및 기본적인 전처리가 진행되고, pod 안에 mean-std.txt가 생성된다.

### hyperparameter tuning
```commandline
model_hyp_train_test : "hyp"
preprocess : ""
model_path : ""
device : "cpu", "0", "1", "2", "3" 중 입력
```
kubeflow 에서 제공하는 Katib도 있으나 참고한 사이트에서 wandb를 이용했고, 실제 UI도 더 마음에 들었다.</BR>
Run 되는 화면에서 logs를 확인하면 접속 가능한 url을 얻을 수 있다.

### train
```commandline
model_hyp_train_test : "train"
preprocess : ""
model_path : ""
device : "cpu", "0", "1", "2", "3" 중 입력
```
pipeline 구성 시 hyperparameter 단계에서 얻어진 best 결과를 전달하도록 설정하려면</BR>
hyperparameter.py 파일 수정 후 도커 Hub에 업데이트가 필요하다.</BR></BR>
여기서 모델 train 시에는 train.py 파일 안에 직접 입력하도록 하였다.

### test
```commandline
model_hyp_train_test : "test"
preprocess : ""
model_path : "mlflow에 업로드 된 s3 주소"
device : "cpu", "0", "1", "2", "3" 중 입력
```
test 수행은 mlflow 에 업로드 되어있는 모델을 가지고 진행된다.</BR>
train 단계를 진행하면 mlflow dashboard에서 모델을 확인할 수 있고 </BR>
거기서 얻은 주소를 run parameter에 넣는다.
#### mlflow 접속
```commandline
kubectl port-forward --address 0.0.0.0 svc/mlflow-server-service -n mlflow-system 8080:5000
```