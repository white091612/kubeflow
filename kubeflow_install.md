# kubeflow 설치 가이드
## 1. minikube 셋업
### docker 컨테이너 생성 및 접속

우리는 도커 컨테이너 내부에 또다시 도커를 설치해서 이용할 예정이므로 최초 컨테이너 설정 시 포트를 설정해줘야 한다.</BR>
Server : 5995   /   Docker : 8080
```
docker run --privileged -v /sys/fs/cgroup:/sys/fs/cgroup:ro -i -t -p 5995:8080 --name joseph.kang_kubeflow ubuntu:focal-20230308
```
### 컨테이너 내 기본 패키지 설치
```
apt-get update
apt-get install sudo
sudo apt-get install ca-certificates curl gnupg lsb-release systemctl nano wget vim
```
### add GPG key
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
### ip 관련 PATH 설정
```nano ~/.bashrc``` 입력 후, 가장 하위로 스크롤하여 아래 줄을 추가한다.</BR>
```export PATH=$PATH:/usr/sbin```</BR></BR>
“Ctrl + X” 를 눌러 에디터를 나오고, “Y” 를 누르면 변경내용이 저장된다.</BR>
```source ~/.bashrc``` 를 입력해서 변경사항을 반영한다.

### install docker engine
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
### check install 
```
sudo docker run hello-world
```
### minikube 1.21.0 설치
```
curl -LO https://storage.googleapis.com/minikube/releases/v1.24.0/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
#### Exiting due to GUEST_MISSING_CONNTRACK: Sorry, Kubernetes 1.20.2 requires conntrack to be installed in root's path 에러가 뜰 경우
```
sudo apt install conntrack
```
### PID1 관련 에러가 무조건 뜨니까 아래 줄 입력
```
sudo apt-get update && sudo apt-get install -yqq daemonize dbus-user-session fontconfig
sudo daemonize /usr/bin/unshare --fork --pid --mount-proc /lib/systemd/systemd --system-unit=basic.target
```
sudo 권한 부여하기
```
nano /etc/sudoers
```
가장 하위에 아래 내용 입력
```
joseph ALL = (ALL:ALL) ALL
```
저장하고 나와서 계속 진행
```
exec sudo nsenter -t $(pidof systemd) -a su - $LOGNAME
```
### root 계정으로는 minikkube 설치가 안되기 때문에 user를 추가합니다.
```
adduser joseph
# Password
usermod –aG sudo joseph
su - joseph
sudo usermod -aG docker $USER && newgrp docker
```
### minikube 클러스터 생성
```
minikube start --driver=docker --cpus 16 --memory=40g --disk-size 40GB --kubernetes-version=1.21.7 --profile=mk
```
### kubectl 설정
미니쿠베에서는 kubectl 명령어를 직접 사용할 수 없어서 다음과 같이 minikube 명령 뒤에 kubectl – 를 추가해서 kubectl 명령을 사용해야 합니다.</BR>
```
minikube kubectl -- <kubectl commands>
```
이렇게 사용하는 건 너무 불편하므로 alias 를 활용해서 minikube 환경에서도 kubectl 명령을 사용할 수 있습니다.</BR>
다음 명령어를 shell 초기화 파일(~/.bashrc, ~/.zshrc) 에 추가해 줍니다.
```
nano ~/.bashrc
```
파일 뜨면 하위에 아래 추가
```
alias kubectl="minikube kubectl --"
```
설정이 끝났으면 반영하기 위해 다시 로그인하거나 설정 파일을 다시 읽기 위해 다음 명령을 실행합니다.
```
source ~/.bashrc
```

### minikube 기본 profile 설정
```
minikube profile mk
```

## kustomize 
kustomize 또한 여러 쿠버네티스 리소스를 한 번에 배포하고 관리할 수 있게 도와주는 패키지 매니징 도구 중 하나입니다.</BR>
현재 폴더에 kustomize v3.10.0 버전의 바이너리를 다운받습니다.</BR>
```
wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv3.10.0/kustomize_v3.10.0_linux_amd64.tar.gz
```
다른 OS는 https://kustomize/v3.10.0에서 확인 후 다운로드 받습니다.</BR></BR>
kustomize 를 사용할 수 있도록 압축을 풀고, 파일의 위치를 변경합니다.</BR>
```
tar -zxvf kustomize_v3.10.0_linux_amd64.tar.gz
sudo mv kustomize /usr/local/bin/kustomize
```
정상적으로 설치되었는지 확인합니다.
```
kustomize help
```
다음과 같은 메시지가 보이면 정상적으로 설치된 것을 의미합니다.
```
Manages declarative configuration of Kubernetes.
See https://sigs.k8s.io/kustomize

Usage:
  kustomize [command]

Available Commands:
  build                     Print configuration per contents of kustomization.yaml
  cfg                       Commands for reading and writing configuration.
  completion                Generate shell completion script
  create                    Create a new kustomization in the current directory
  edit                      Edits a kustomization file
  fn                        Commands for running functions against configuration.
```
### CSI Plugin : Local Path Provisioner 
CSI Plugin은 kubernetes 내의 스토리지를 담당하는 모듈입니다. </BR>
단일 노드 클러스터에서 쉽게 사용할 수 있는 CSI Plugin인 Local Path Provisioner를 설치합니다.
```
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.20/deploy/local-path-storage.yaml
```

다음과 같은 메시지가 보이면 정상적으로 설치된 것을 의미합니다.
```
namespace/local-path-storage created
serviceaccount/local-path-provisioner-service-account created
clusterrole.rbac.authorization.k8s.io/local-path-provisioner-role created
clusterrolebinding.rbac.authorization.k8s.io/local-path-provisioner-bind created
deployment.apps/local-path-provisioner created
storageclass.storage.k8s.io/local-path created
configmap/local-path-config created
```

또한, 다음과 같이 local-path-storage namespace 에 provisioner pod이 Running 인지 확인합니다.
```
kubectl -n local-path-storage get pod
```

정상적으로 수행되면 아래와 같이 출력됩니다.
```
NAME                                     READY     STATUS    RESTARTS   AGE
local-path-provisioner-d744ccf98-xfcbk   1/1       Running   0          7m
```

다음을 수행하여 default storage class로 변경합니다.
```
kubectl patch storageclass local-path  -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

정상적으로 수행되면 아래와 같이 출력됩니다.</BR>
</BR>
```
storageclass.storage.k8s.io/local-path patched
```

default storage class로 설정되었는지 확인합니다.
```
kubectl get sc
```

다음과 같이 NAME에 local-path (default) 인 storage class가 존재하는 것을 확인합니다.
```
NAME                   PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
local-path (default)   rancher.io/local-path   Delete          WaitForFirstConsumer   false                  2h
```
default 가 2개 이상인 경우 아래 명령어로 1개만 default로 설정되도록 합니다.</BR>
(NAME 이 standard가 아닌경우 다른걸로 변경해서 입력)
```
kubectl patch sc standard -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "false"}}}'
```

### 정상 설치 확인 
최종적으로 node가 Ready 인지, OS, Docker, Kubernetes 버전을 확인합니다.
```
kubectl get nodes -o wide
```
다음과 같은 메시지가 보이면 정상적으로 설치된 것을 의미합니다.
```
NAME     STATUS   ROLES                  AGE     VERSION   INTERNAL-IP    EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION     CONTAINER-RUNTIME
mk       Ready    control-plane,master   2d23h   v1.21.7   192.168.0.75   <none>        Ubuntu 20.04.3 LTS   5.4.0
```


## Kubeflow
https://mlops-for-all.github.io/en/docs/setup-components/install-components-kf/</BR>
위 사이트 보고 진행했음</BR>
### 설치 파일 준비 
Kubeflow v1.4.0 버전을 설치하기 위해서, 설치에 필요한 manifests 파일들을 준비합니다.</BR>
kubeflow/manifests Repository 를 v1.4.0 태그로 깃 클론한 뒤, 해당 폴더로 이동합니다.
```
git clone -b v1.4.0 https://github.com/kubeflow/manifests.git
cd manifests
```
### cert-manager

1. cert-manager 를 설치합니다.

  ```text
  kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -
  ```

  cert-manager namespace 의 3 개의 pod 가 모두 Running 이 될 때까지 기다립니다.

  ```text
  kubectl get pod -n cert-manager
  ```

  모두 Running 이 되면 다음과 비슷한 결과가 출력됩니다.

  ```text
  NAME                                       READY   STATUS    RESTARTS   AGE
  cert-manager-7dd5854bb4-7nmpd              1/1     Running   0          2m10s
  cert-manager-cainjector-64c949654c-2scxr   1/1     Running   0          2m10s
  cert-manager-webhook-6b57b9b886-7q6g2      1/1     Running   0          2m10s
  ```
  2. 사이트에 있는거로 하면 안되고 이거로 해야함
```
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.3.0/cert-manager.yaml
```
정상적으로 설치되면 다음과 같이 출력됩니다.

  ```text
  namespace/cert-manager created
  customresourcedefinition.apiextensions.k8s.io/certificaterequests.cert-manager.io created
  customresourcedefinition.apiextensions.k8s.io/certificates.cert-manager.io created
  customresourcedefinition.apiextensions.k8s.io/challenges.acme.cert-manager.io created
  customresourcedefinition.apiextensions.k8s.io/clusterissuers.cert-manager.io created
  customresourcedefinition.apiextensions.k8s.io/issuers.cert-manager.io created
  customresourcedefinition.apiextensions.k8s.io/orders.acme.cert-manager.io created
  serviceaccount/cert-manager created
  serviceaccount/cert-manager-cainjector created
  serviceaccount/cert-manager-webhook created
  role.rbac.authorization.k8s.io/cert-manager-webhook:dynamic-serving created
  role.rbac.authorization.k8s.io/cert-manager-cainjector:leaderelection created
  role.rbac.authorization.k8s.io/cert-manager:leaderelection created
  clusterrole.rbac.authorization.k8s.io/cert-manager-cainjector created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-approve:cert-manager-io created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-certificates created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-challenges created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-clusterissuers created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-ingress-shim created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-issuers created
  clusterrole.rbac.authorization.k8s.io/cert-manager-controller-orders created
  clusterrole.rbac.authorization.k8s.io/cert-manager-edit created
  clusterrole.rbac.authorization.k8s.io/cert-manager-view created
  clusterrole.rbac.authorization.k8s.io/cert-manager-webhook:subjectaccessreviews created
  rolebinding.rbac.authorization.k8s.io/cert-manager-webhook:dynamic-serving created
  rolebinding.rbac.authorization.k8s.io/cert-manager-cainjector:leaderelection created
  rolebinding.rbac.authorization.k8s.io/cert-manager:leaderelection created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-cainjector created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-approve:cert-manager-io created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-certificates created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-challenges created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-clusterissuers created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-ingress-shim created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-issuers created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-orders created
  clusterrolebinding.rbac.authorization.k8s.io/cert-manager-webhook:subjectaccessreviews created
  service/cert-manager created
  service/cert-manager-webhook created
  deployment.apps/cert-manager created
  deployment.apps/cert-manager-cainjector created
  deployment.apps/cert-manager-webhook created
  mutatingwebhookconfiguration.admissionregistration.k8s.io/cert-manager-webhook created
  validatingwebhookconfiguration.admissionregistration.k8s.io/cert-manager-webhook created
  ```

- cert-manager-webhook 이슈

  cert-manager-webhook deployment 가 Running 이 아닌 경우, 다음과 비슷한 에러가 발생하며 kubeflow-issuer가 설치되지 않을 수 있음에 주의하시기 바랍니다.  
  해당 에러가 발생한 경우, cert-manager 의 3개의 pod 가 모두 Running 이 되는 것을 확인한 이후 다시 명령어를 수행하시기 바랍니다.

  ```text
  Error from server: error when retrieving current configuration of:
  Resource: "cert-manager.io/v1alpha2, Resource=clusterissuers", GroupVersionKind: "cert-manager.io/v1alpha2, Kind=ClusterIssuer"
  Name: "kubeflow-self-signing-issuer", Namespace: ""
  from server for: "STDIN": conversion webhook for cert-manager.io/v1, Kind=ClusterIssuer failed: Post "https://cert-manager-webhook.cert-manager.svc:443/convert?timeout=30s": dial tcp 10.101.177.157:443: connect: connection refused
  ```

### Istio

1. istio 관련 Custom Resource Definition(CRD) 를 설치합니다.

  ```text
  kustomize build common/istio-1-9/istio-crds/base | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  customresourcedefinition.apiextensions.k8s.io/authorizationpolicies.security.istio.io created
  customresourcedefinition.apiextensions.k8s.io/destinationrules.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/envoyfilters.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/gateways.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/istiooperators.install.istio.io created
  customresourcedefinition.apiextensions.k8s.io/peerauthentications.security.istio.io created
  customresourcedefinition.apiextensions.k8s.io/requestauthentications.security.istio.io created
  customresourcedefinition.apiextensions.k8s.io/serviceentries.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/sidecars.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/virtualservices.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/workloadentries.networking.istio.io created
  customresourcedefinition.apiextensions.k8s.io/workloadgroups.networking.istio.io created
  ```

2. istio namespace 를 설치합니다.

  ```text
  kustomize build common/istio-1-9/istio-namespace/base | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  namespace/istio-system created
  ```

3. istio 를 설치합니다.

  ```text
  kustomize build common/istio-1-9/istio-install/base | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  serviceaccount/istio-ingressgateway-service-account created
  serviceaccount/istio-reader-service-account created
  serviceaccount/istiod-service-account created
  role.rbac.authorization.k8s.io/istio-ingressgateway-sds created
  role.rbac.authorization.k8s.io/istiod-istio-system created
  clusterrole.rbac.authorization.k8s.io/istio-reader-istio-system created
  clusterrole.rbac.authorization.k8s.io/istiod-istio-system created
  rolebinding.rbac.authorization.k8s.io/istio-ingressgateway-sds created
  rolebinding.rbac.authorization.k8s.io/istiod-istio-system created
  clusterrolebinding.rbac.authorization.k8s.io/istio-reader-istio-system created
  clusterrolebinding.rbac.authorization.k8s.io/istiod-istio-system created
  configmap/istio created
  configmap/istio-sidecar-injector created
  service/istio-ingressgateway created
  service/istiod created
  deployment.apps/istio-ingressgateway created
  deployment.apps/istiod created
  envoyfilter.networking.istio.io/metadata-exchange-1.8 created
  envoyfilter.networking.istio.io/metadata-exchange-1.9 created
  envoyfilter.networking.istio.io/stats-filter-1.8 created
  envoyfilter.networking.istio.io/stats-filter-1.9 created
  envoyfilter.networking.istio.io/tcp-metadata-exchange-1.8 created
  envoyfilter.networking.istio.io/tcp-metadata-exchange-1.9 created
  envoyfilter.networking.istio.io/tcp-stats-filter-1.8 created
  envoyfilter.networking.istio.io/tcp-stats-filter-1.9 created
  envoyfilter.networking.istio.io/x-forwarded-host created
  gateway.networking.istio.io/istio-ingressgateway created
  authorizationpolicy.security.istio.io/global-deny-all created
  authorizationpolicy.security.istio.io/istio-ingressgateway created
  mutatingwebhookconfiguration.admissionregistration.k8s.io/istio-sidecar-injector created
  validatingwebhookconfiguration.admissionregistration.k8s.io/istiod-istio-system created
  ```

  istio-system namespace 의 2 개의 pod 가 모두 Running 이 될 때까지 기다립니다.

  ```text
  kubectl get po -n istio-system
  ```

  모두 Running 이 되면 다음과 비슷한 결과가 출력됩니다.

  ```text
  NAME                                   READY   STATUS    RESTARTS   AGE
  istio-ingressgateway-79b665c95-xm22l   1/1     Running   0          16s
  istiod-86457659bb-5h58w                1/1     Running   0          16s
  ```

### Dex

dex 를 설치합니다.

```text
kustomize build common/dex/overlays/istio | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
namespace/auth created
customresourcedefinition.apiextensions.k8s.io/authcodes.dex.coreos.com created
serviceaccount/dex created
clusterrole.rbac.authorization.k8s.io/dex created
clusterrolebinding.rbac.authorization.k8s.io/dex created
configmap/dex created
secret/dex-oidc-client created
service/dex created
deployment.apps/dex created
virtualservice.networking.istio.io/dex created
```

auth namespace 의 1 개의 pod 가 모두 Running 이 될 때까지 기다립니다.

```text
kubectl get po -n auth
```

모두 Running 이 되면 다음과 비슷한 결과가 출력됩니다.

```text
NAME                   READY   STATUS    RESTARTS   AGE
dex-5ddf47d88d-458cs   1/1     Running   1          12s
```

### OIDC AuthService

OIDC AuthService 를 설치합니다.

```text
kustomize build common/oidc-authservice/base | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
configmap/oidc-authservice-parameters created
secret/oidc-authservice-client created
service/authservice created
persistentvolumeclaim/authservice-pvc created
statefulset.apps/authservice created
envoyfilter.networking.istio.io/authn-filter created
```

istio-system namespace 에 authservice-0 pod 가 Running 이 될 때까지 기다립니다.

```text
kubectl get po -n istio-system -w
```

모두 Running 이 되면 다음과 비슷한 결과가 출력됩니다.

```text
NAME                                   READY   STATUS    RESTARTS   AGE
authservice-0                          1/1     Running   0          14s
istio-ingressgateway-79b665c95-xm22l   1/1     Running   0          2m37s
istiod-86457659bb-5h58w                1/1     Running   0          2m37s
```

### Kubeflow Namespace

kubeflow namespace 를 생성합니다.

```text
kustomize build common/kubeflow-namespace/base | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
namespace/kubeflow created
```

kubeflow namespace 를 조회합니다.

```text
kubectl get ns kubeflow
```

정상적으로 생성되면 다음과 비슷한 결과가 출력됩니다.

```text
NAME       STATUS   AGE
kubeflow   Active   8s
```

### Kubeflow Roles

kubeflow-roles 를 설치합니다.

```text
kustomize build common/kubeflow-roles/base | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
clusterrole.rbac.authorization.k8s.io/kubeflow-admin created
clusterrole.rbac.authorization.k8s.io/kubeflow-edit created
clusterrole.rbac.authorization.k8s.io/kubeflow-kubernetes-admin created
clusterrole.rbac.authorization.k8s.io/kubeflow-kubernetes-edit created
clusterrole.rbac.authorization.k8s.io/kubeflow-kubernetes-view created
clusterrole.rbac.authorization.k8s.io/kubeflow-view created
```

방금 생성한 kubeflow roles 를 조회합니다.

```text
kubectl get clusterrole | grep kubeflow
```

다음과 같이 총 6개의 clusterrole 이 출력됩니다.

```text
kubeflow-admin                                                         2021-12-03T08:51:36Z
kubeflow-edit                                                          2021-12-03T08:51:36Z
kubeflow-kubernetes-admin                                              2021-12-03T08:51:36Z
kubeflow-kubernetes-edit                                               2021-12-03T08:51:36Z
kubeflow-kubernetes-view                                               2021-12-03T08:51:36Z
kubeflow-view                                                          2021-12-03T08:51:36Z
```

### Kubeflow Istio Resources

kubeflow-istio-resources 를 설치합니다.

```text
kustomize build common/istio-1-9/kubeflow-istio-resources/base | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
clusterrole.rbac.authorization.k8s.io/kubeflow-istio-admin created
clusterrole.rbac.authorization.k8s.io/kubeflow-istio-edit created
clusterrole.rbac.authorization.k8s.io/kubeflow-istio-view created
gateway.networking.istio.io/kubeflow-gateway created
```

방금 생성한 kubeflow roles 를 조회합니다.

```text
kubectl get clusterrole | grep kubeflow-istio
```

다음과 같이 총 3개의 clusterrole 이 출력됩니다.

```text
kubeflow-istio-admin                                                   2021-12-03T08:53:17Z
kubeflow-istio-edit                                                    2021-12-03T08:53:17Z
kubeflow-istio-view                                                    2021-12-03T08:53:17Z
```

Kubeflow namespace 에 gateway 가 정상적으로 설치되었는지 확인합니다.

```text
kubectl get gateway -n kubeflow
```

정상적으로 생성되면 다음과 비슷한 결과가 출력됩니다.

```text
NAME               AGE
kubeflow-gateway   31s
```

### Kubeflow Pipelines

kubeflow pipelines 를 설치합니다.

```text
kustomize build apps/pipeline/upstream/env/platform-agnostic-multi-user | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
customresourcedefinition.apiextensions.k8s.io/clusterworkflowtemplates.argoproj.io created
customresourcedefinition.apiextensions.k8s.io/cronworkflows.argoproj.io created
customresourcedefinition.apiextensions.k8s.io/workfloweventbindings.argoproj.io created
...(생략)
authorizationpolicy.security.istio.io/ml-pipeline-visualizationserver created
authorizationpolicy.security.istio.io/mysql created
authorizationpolicy.security.istio.io/service-cache-server created
```

위 명령어는 여러 resources 를 한 번에 설치하고 있지만, 설치 순서의 의존성이 있는 리소스가 존재합니다.  
따라서 때에 따라 다음과 비슷한 에러가 발생할 수 있습니다.

```text
"error: unable to recognize "STDIN": no matches for kind "CompositeController" in version "metacontroller.k8s.io/v1alpha1""  
```

위와 비슷한 에러가 발생한다면, 10 초 정도 기다린 뒤 다시 위의 명령을 수행합니다.

```text
kustomize build apps/pipeline/upstream/env/platform-agnostic-multi-user | kubectl apply -f -
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow
```

다음과 같이 총 16개의 pod 가 모두 Running 이 될 때까지 기다립니다.

```text
NAME                                                     READY   STATUS    RESTARTS   AGE
cache-deployer-deployment-79fdf9c5c9-bjnbg               2/2     Running   1          5m3s
cache-server-5bdf4f4457-48gbp                            2/2     Running   0          5m3s
kubeflow-pipelines-profile-controller-7b947f4748-8d26b   1/1     Running   0          5m3s
metacontroller-0                                         1/1     Running   0          5m3s
metadata-envoy-deployment-5b4856dd5-xtlkd                1/1     Running   0          5m3s
metadata-grpc-deployment-6b5685488-kwvv7                 2/2     Running   3          5m3s
metadata-writer-548bd879bb-zjkcn                         2/2     Running   1          5m3s
minio-5b65df66c9-k5gzg                                   2/2     Running   0          5m3s
ml-pipeline-8c4b99589-85jw6                              2/2     Running   1          5m3s
ml-pipeline-persistenceagent-d6bdc77bd-ssxrv             2/2     Running   0          5m3s
ml-pipeline-scheduledworkflow-5db54d75c5-zk2cw           2/2     Running   0          5m2s
ml-pipeline-ui-5bd8d6dc84-j7wqr                          2/2     Running   0          5m2s
ml-pipeline-viewer-crd-68fb5f4d58-mbcbg                  2/2     Running   1          5m2s
ml-pipeline-visualizationserver-8476b5c645-wljfm         2/2     Running   0          5m2s
mysql-f7b9b7dd4-xfnw4                                    2/2     Running   0          5m2s
workflow-controller-5cbbb49bd8-5zrwx                     2/2     Running   1          5m2s
```

추가로 ml-pipeline UI가 정상적으로 접속되는지 확인합니다.

```text
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

웹 브라우저를 열어 [http://ServerIP:5995/#/pipelines/](http://ServerIP:5995/#/pipelines/) 경로에 접속합니다.

다음과 같은 화면이 출력되는 것을 확인합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/231660660-0c5f73c5-e0ae-4072-ba62-3c9bff10d8b9.png" title="pipeline-ui"/>
</p>

- localhost 연결 거부 이슈

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/231660640-3c828bc9-244d-4074-b0ae-5128175ab8cc.png" title="localhost-reject"/>
</p>

만약 다음과 같이 `localhost에서 연결을 거부했습니다` 라는 에러가 출력될 경우, 커맨드로 address 설정을 통해 접근하는 것이 가능합니다.

**보안상의 문제가 되지 않는다면,** 아래와 같이 `0.0.0.0` 로 모든 주소의 bind를 열어주는 방향으로 ml-pipeline UI가 정상적으로 접속되는지 확인합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/ml-pipeline-ui -n kubeflow 8080:80
```

- 위의 옵션으로 실행했음에도 여전히 연결 거부 이슈가 발생할 경우

방화벽 설정으로 접속해 모든 tcp 프로토콜의 포트에 대한 접속을 허가 또는 8888번 포트의 접속 허가를 추가해 접근 권한을 허가해줍니다.

웹 브라우저를 열어 `http://<당신의 가상 인스턴스 공인 ip 주소>:5995/#/pipelines/` 경로에 접속하면, ml-pipeline UI 화면이 출력되는 것을 확인할 수 있습니다.

하단에서 진행되는 다른 포트의 경로에 접속할 때도 위의 절차와 동일하게 커맨드를 실행하고, 방화벽에 포트 번호를 추가해주면 실행하는 것이 가능합니다.

### Katib

Katib 를 설치합니다.

```text
kustomize build apps/katib/upstream/installs/katib-with-kubeflow | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
customresourcedefinition.apiextensions.k8s.io/experiments.kubeflow.org created
customresourcedefinition.apiextensions.k8s.io/suggestions.kubeflow.org created
customresourcedefinition.apiextensions.k8s.io/trials.kubeflow.org created
serviceaccount/katib-controller created
serviceaccount/katib-ui created
clusterrole.rbac.authorization.k8s.io/katib-controller created
clusterrole.rbac.authorization.k8s.io/katib-ui created
clusterrole.rbac.authorization.k8s.io/kubeflow-katib-admin created
clusterrole.rbac.authorization.k8s.io/kubeflow-katib-edit created
clusterrole.rbac.authorization.k8s.io/kubeflow-katib-view created
clusterrolebinding.rbac.authorization.k8s.io/katib-controller created
clusterrolebinding.rbac.authorization.k8s.io/katib-ui created
configmap/katib-config created
configmap/trial-templates created
secret/katib-mysql-secrets created
service/katib-controller created
service/katib-db-manager created
service/katib-mysql created
service/katib-ui created
persistentvolumeclaim/katib-mysql created
deployment.apps/katib-controller created
deployment.apps/katib-db-manager created
deployment.apps/katib-mysql created
deployment.apps/katib-ui created
certificate.cert-manager.io/katib-webhook-cert created
issuer.cert-manager.io/katib-selfsigned-issuer created
virtualservice.networking.istio.io/katib-ui created
mutatingwebhookconfiguration.admissionregistration.k8s.io/katib.kubeflow.org created
validatingwebhookconfiguration.admissionregistration.k8s.io/katib.kubeflow.org created
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow | grep katib
```

다음과 같이 총 4 개의 pod 가 Running 이 될 때까지 기다립니다.

```text
katib-controller-68c47fbf8b-b985z                        1/1     Running   0          82s
katib-db-manager-6c948b6b76-2d9gr                        1/1     Running   0          82s
katib-mysql-7894994f88-scs62                             1/1     Running   0          82s
katib-ui-64bb96d5bf-d89kp                                1/1     Running   0          82s
```

추가로 katib UI가 정상적으로 접속되는지 확인합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/katib-ui -n kubeflow 8080:80
```

웹 브라우저를 열어 [http://ServerIP:5995/katib/](http://ServerIP:5995/katib/) 경로에 접속합니다.

다음과 같은 화면이 출력되는 것을 확인합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/231660617-90e2fe48-c689-4783-85c3-63a96b08b18f.png" title="katib-ui"/>
</p>

### Central Dashboard

Dashboard 를 설치합니다.

```text
kustomize build apps/centraldashboard/upstream/overlays/istio | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
serviceaccount/centraldashboard created
role.rbac.authorization.k8s.io/centraldashboard created
clusterrole.rbac.authorization.k8s.io/centraldashboard created
rolebinding.rbac.authorization.k8s.io/centraldashboard created
clusterrolebinding.rbac.authorization.k8s.io/centraldashboard created
configmap/centraldashboard-config created
configmap/centraldashboard-parameters created
service/centraldashboard created
deployment.apps/centraldashboard created
virtualservice.networking.istio.io/centraldashboard created
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow | grep centraldashboard
```

kubeflow namespace 에 centraldashboard 관련 1 개의 pod 가 Running 이 될 때까지 기다립니다.

```text
centraldashboard-8fc7d8cc-xl7ts                          1/1     Running   0          52s
```

추가로 Central Dashboard UI가 정상적으로 접속되는지 확인합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/centraldashboard -n kubeflow 8080:80
```

웹 브라우저를 열어 [http://ServerIP:5995/](http://ServerIP:5995/) 경로에 접속합니다.

다음과 같은 화면이 출력되는 것을 확인합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/231660606-92986c4a-c580-482c-b939-ad39bb8f1a41.png" title="central-dashboard"/>
</p>

### Admission Webhook

```text
kustomize build apps/admission-webhook/upstream/overlays/cert-manager | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
customresourcedefinition.apiextensions.k8s.io/poddefaults.kubeflow.org created
serviceaccount/admission-webhook-service-account created
clusterrole.rbac.authorization.k8s.io/admission-webhook-cluster-role created
clusterrole.rbac.authorization.k8s.io/admission-webhook-kubeflow-poddefaults-admin created
clusterrole.rbac.authorization.k8s.io/admission-webhook-kubeflow-poddefaults-edit created
clusterrole.rbac.authorization.k8s.io/admission-webhook-kubeflow-poddefaults-view created
clusterrolebinding.rbac.authorization.k8s.io/admission-webhook-cluster-role-binding created
service/admission-webhook-service created
deployment.apps/admission-webhook-deployment created
certificate.cert-manager.io/admission-webhook-cert created
issuer.cert-manager.io/admission-webhook-selfsigned-issuer created
mutatingwebhookconfiguration.admissionregistration.k8s.io/admission-webhook-mutating-webhook-configuration created
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow | grep admission-webhook
```

1 개의 pod 가 Running 이 될 때까지 기다립니다.

```text
admission-webhook-deployment-667bd68d94-2hhrx            1/1     Running   0          11s
```

### Notebooks & Jupyter Web App

1. Notebook controller 를 설치합니다.

  ```text
  kustomize build apps/jupyter/notebook-controller/upstream/overlays/kubeflow | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  customresourcedefinition.apiextensions.k8s.io/notebooks.kubeflow.org created
  serviceaccount/notebook-controller-service-account created
  role.rbac.authorization.k8s.io/notebook-controller-leader-election-role created
  clusterrole.rbac.authorization.k8s.io/notebook-controller-kubeflow-notebooks-admin created
  clusterrole.rbac.authorization.k8s.io/notebook-controller-kubeflow-notebooks-edit created
  clusterrole.rbac.authorization.k8s.io/notebook-controller-kubeflow-notebooks-view created
  clusterrole.rbac.authorization.k8s.io/notebook-controller-role created
  rolebinding.rbac.authorization.k8s.io/notebook-controller-leader-election-rolebinding created
  clusterrolebinding.rbac.authorization.k8s.io/notebook-controller-role-binding created
  configmap/notebook-controller-config-m44cmb547t created
  service/notebook-controller-service created
  deployment.apps/notebook-controller-deployment created
  ```

  정상적으로 설치되었는지 확인합니다.

  ```text
  kubectl get po -n kubeflow | grep notebook-controller
  ```

  1 개의 pod 가 Running 이 될 때까지 기다립니다.

  ```text
  notebook-controller-deployment-75b4f7b578-w4d4l          1/1     Running   0          105s
  ```

2. Jupyter Web App 을 설치합니다.

  ```text
  kustomize build apps/jupyter/jupyter-web-app/upstream/overlays/istio | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  serviceaccount/jupyter-web-app-service-account created
  role.rbac.authorization.k8s.io/jupyter-web-app-jupyter-notebook-role created
  clusterrole.rbac.authorization.k8s.io/jupyter-web-app-cluster-role created
  clusterrole.rbac.authorization.k8s.io/jupyter-web-app-kubeflow-notebook-ui-admin created
  clusterrole.rbac.authorization.k8s.io/jupyter-web-app-kubeflow-notebook-ui-edit created
  clusterrole.rbac.authorization.k8s.io/jupyter-web-app-kubeflow-notebook-ui-view created
  rolebinding.rbac.authorization.k8s.io/jupyter-web-app-jupyter-notebook-role-binding created
  clusterrolebinding.rbac.authorization.k8s.io/jupyter-web-app-cluster-role-binding created
  configmap/jupyter-web-app-config-76844k4cd7 created
  configmap/jupyter-web-app-logos created
  configmap/jupyter-web-app-parameters-chmg88cm48 created
  service/jupyter-web-app-service created
  deployment.apps/jupyter-web-app-deployment created
  virtualservice.networking.istio.io/jupyter-web-app-jupyter-web-app created
  ```

  정상적으로 설치되었는지 확인합니다.

  ```text
  kubectl get po -n kubeflow | grep jupyter-web-app
  ```

  1개의 pod 가 Running 이 될 때까지 기다립니다.

  ```text
  jupyter-web-app-deployment-6f744fbc54-p27ts              1/1     Running   0          2m
  ```

### Profiles + KFAM

Profile Controller를 설치합니다.

```text
kustomize build apps/profiles/upstream/overlays/kubeflow | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
customresourcedefinition.apiextensions.k8s.io/profiles.kubeflow.org created
serviceaccount/profiles-controller-service-account created
role.rbac.authorization.k8s.io/profiles-leader-election-role created
rolebinding.rbac.authorization.k8s.io/profiles-leader-election-rolebinding created
clusterrolebinding.rbac.authorization.k8s.io/profiles-cluster-role-binding created
configmap/namespace-labels-data-48h7kd55mc created
configmap/profiles-config-46c7tgh6fd created
service/profiles-kfam created
deployment.apps/profiles-deployment created
virtualservice.networking.istio.io/profiles-kfam created
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow | grep profiles-deployment
```

1 개의 pod 가 Running 이 될 때까지 기다립니다.

```text
profiles-deployment-89f7d88b-qsnrd                       2/2     Running   0          42s
```

### Volumes Web App

Volumes Web App 을 설치합니다.

```text
kustomize build apps/volumes-web-app/upstream/overlays/istio | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
serviceaccount/volumes-web-app-service-account created
clusterrole.rbac.authorization.k8s.io/volumes-web-app-cluster-role created
clusterrole.rbac.authorization.k8s.io/volumes-web-app-kubeflow-volume-ui-admin created
clusterrole.rbac.authorization.k8s.io/volumes-web-app-kubeflow-volume-ui-edit created
clusterrole.rbac.authorization.k8s.io/volumes-web-app-kubeflow-volume-ui-view created
clusterrolebinding.rbac.authorization.k8s.io/volumes-web-app-cluster-role-binding created
configmap/volumes-web-app-parameters-4gg8cm2gmk created
service/volumes-web-app-service created
deployment.apps/volumes-web-app-deployment created
virtualservice.networking.istio.io/volumes-web-app-volumes-web-app created
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow | grep volumes-web-app
```

1개의 pod가 Running 이 될 때까지 기다립니다.

```text
volumes-web-app-deployment-8589d664cc-62svl              1/1     Running   0          27s
```

### Tensorboard & Tensorboard Web App

1. Tensorboard Web App 를 설치합니다.

  ```text
  kustomize build apps/tensorboard/tensorboards-web-app/upstream/overlays/istio | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  serviceaccount/tensorboards-web-app-service-account created
  clusterrole.rbac.authorization.k8s.io/tensorboards-web-app-cluster-role created
  clusterrole.rbac.authorization.k8s.io/tensorboards-web-app-kubeflow-tensorboard-ui-admin created
  clusterrole.rbac.authorization.k8s.io/tensorboards-web-app-kubeflow-tensorboard-ui-edit created
  clusterrole.rbac.authorization.k8s.io/tensorboards-web-app-kubeflow-tensorboard-ui-view created
  clusterrolebinding.rbac.authorization.k8s.io/tensorboards-web-app-cluster-role-binding created
  configmap/tensorboards-web-app-parameters-g28fbd6cch created
  service/tensorboards-web-app-service created
  deployment.apps/tensorboards-web-app-deployment created
  virtualservice.networking.istio.io/tensorboards-web-app-tensorboards-web-app created
  ```

  정상적으로 설치되었는지 확인합니다.

  ```text
  kubectl get po -n kubeflow | grep tensorboards-web-app
  ```

  1 개의 pod 가 Running 이 될 때까지 기다립니다.

  ```text
  tensorboards-web-app-deployment-6ff79b7f44-qbzmw            1/1     Running             0          22s
  ```

2. Tensorboard Controller 를 설치합니다.

  ```text
  kustomize build apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow | kubectl apply -f -
  ```

  정상적으로 수행되면 다음과 같이 출력됩니다.

  ```text
  customresourcedefinition.apiextensions.k8s.io/tensorboards.tensorboard.kubeflow.org created
  serviceaccount/tensorboard-controller created
  role.rbac.authorization.k8s.io/tensorboard-controller-leader-election-role created
  clusterrole.rbac.authorization.k8s.io/tensorboard-controller-manager-role created
  clusterrole.rbac.authorization.k8s.io/tensorboard-controller-proxy-role created
  rolebinding.rbac.authorization.k8s.io/tensorboard-controller-leader-election-rolebinding created
  clusterrolebinding.rbac.authorization.k8s.io/tensorboard-controller-manager-rolebinding created
  clusterrolebinding.rbac.authorization.k8s.io/tensorboard-controller-proxy-rolebinding created
  configmap/tensorboard-controller-config-bf88mm96c8 created
  service/tensorboard-controller-controller-manager-metrics-service created
  deployment.apps/tensorboard-controller-controller-manager created
  ```

  정상적으로 설치되었는지 확인합니다.

  ```text
  kubectl get po -n kubeflow | grep tensorboard-controller
  ```

  1 개의 pod 가 Running 이 될 때까지 기다립니다.

  ```text
  tensorboard-controller-controller-manager-954b7c544-vjpzj   3/3     Running   1          73s
  ```

### Training Operator

Training Operator 를 설치합니다.

```text
kustomize build apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
customresourcedefinition.apiextensions.k8s.io/mxjobs.kubeflow.org created
customresourcedefinition.apiextensions.k8s.io/pytorchjobs.kubeflow.org created
customresourcedefinition.apiextensions.k8s.io/tfjobs.kubeflow.org created
customresourcedefinition.apiextensions.k8s.io/xgboostjobs.kubeflow.org created
serviceaccount/training-operator created
clusterrole.rbac.authorization.k8s.io/kubeflow-training-admin created
clusterrole.rbac.authorization.k8s.io/kubeflow-training-edit created
clusterrole.rbac.authorization.k8s.io/kubeflow-training-view created
clusterrole.rbac.authorization.k8s.io/training-operator created
clusterrolebinding.rbac.authorization.k8s.io/training-operator created
service/training-operator created
deployment.apps/training-operator created
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get po -n kubeflow | grep training-operator
```

1 개의 pod 가 Running 이 될 때까지 기다립니다.

```text
training-operator-7d98f9dd88-6887f                          1/1     Running   0          28s
```

### User Namespace

Kubeflow 사용을 위해, 사용할 User의 Kubeflow Profile 을 생성합니다.

```text
kustomize build common/user-namespace/base | kubectl apply -f -
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
configmap/default-install-config-9h2h2b6hbk created
profile.kubeflow.org/kubeflow-user-example-com created
```

kubeflow-user-example-com profile 이 생성된 것을 확인합니다.

```text
kubectl get profile
```

```text
kubeflow-user-example-com   37s
```

## 정상 설치 확인

Kubeflow central dashboard에 web browser로 접속하기 위해 포트 포워딩합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/istio-ingressgateway -n istio-system 8080:80
```

Web Browser 를 열어 [http://ServerIP:5995](http://ServerIP:5995) 으로 접속하여, 다음과 같은 화면이 출력되는 것을 확인합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/231660575-23d7d711-fb8e-4ff3-bea0-a3e7e1c0f359.png" title="login-ui"/>
</p>

다음 접속 정보를 입력하여 접속합니다.

- Email Address: `user@example.com`
- Password: `12341234`

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/231660365-1a23567c-9072-4021-b0fa-866a0840c494.png" title="central-dashboard"/>
</p>

## Install MLflow Tracking Server

MLflow는 대표적인 오픈소스 ML 실험 관리 도구입니다. MLflow는 [실험 관리 용도](https://mlflow.org/docs/latest/tracking.html#tracking) 외에도 [ML Model 패키징](https://mlflow.org/docs/latest/projects.html#projects), [ML 모델 배포 관리](https://mlflow.org/docs/latest/models.html#models), [ML 모델 저장](https://mlflow.org/docs/latest/model-registry.html#registry)과 같은 기능도 제공하고 있습니다.

*모두의 MLOps*에서는 MLflow를 실험 관리 용도로 사용합니다.  
그래서 MLflow에서 관리하는 데이터를 저장하고 UI를 제공하는 MLflow Tracking Server를 쿠버네티스 클러스터에 배포하여 사용할 예정입니다.

## Before Install MLflow Tracking Server

### PostgreSQL DB 설치

MLflow Tracking Server가 Backend Store로 사용할 용도의 PostgreSQL DB를 쿠버네티스 클러스터에 배포합니다.

먼저 `mlflow-system`이라는 namespace 를 생성합니다.

```text
kubectl create ns mlflow-system
```

다음과 같은 메시지가 출력되면 정상적으로 생성된 것을 의미합니다.

```text
namespace/mlflow-system created
```

postgresql DB를 `mlflow-system` namespace 에 생성합니다.

```text
kubectl -n mlflow-system apply -f https://raw.githubusercontent.com/mlops-for-all/helm-charts/b94b5fe4133f769c04b25068b98ccfa7a505aa60/mlflow/manifests/postgres.yaml 
```

정상적으로 수행되면 다음과 같이 출력됩니다.

```text
service/postgresql-mlflow-service created
deployment.apps/postgresql-mlflow created
persistentvolumeclaim/postgresql-mlflow-pvc created
```

mlflow-system namespace 에 1개의 postgresql 관련 pod 가 Running 이 될 때까지 기다립니다.

```text
kubectl get pod -n mlflow-system | grep postgresql
```

다음과 비슷하게 출력되면 정상적으로 실행된 것입니다.

```text
postgresql-mlflow-7b9bc8c79f-srkh7   1/1     Running   0          38s
```

### Minio 설정

MLflow Tracking Server가 Artifacts Store로 사용할 용도의 Minio는 이전 Kubeflow 설치 단계에서 설치한 Minio를 활용합니다.  
단, kubeflow 용도와 mlflow 용도를 분리하기 위해, mlflow 전용 버킷(bucket)을 생성하겠습니다.  
minio 에 접속하여 버킷을 생성하기 위해, 우선 minio-service 를 포트포워딩합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/minio-service -n kubeflow 8080:9000
```

웹 브라우저를 열어 [ServerIP:5995](http://ServerIP:5995)으로 접속하면 다음과 같은 화면이 출력됩니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232348572-4885e111-353c-4a88-8465-2683487e7eaa.png" title="minio-install"/>
</p>

다음과 같은 접속 정보를 입력하여 로그인합니다.

- Username: `minio`
- Password: `minio123`

우측 하단의 **`+`** 버튼을 클릭하여, `Create Bucket`를 클릭합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232348582-bab7fab5-ed27-49c7-b198-91a7bfb59c7e.png" title="create-bucket"/>
</p>

`Bucket Name`에 `mlflow`를 입력하여 버킷을 생성합니다.

정상적으로 생성되면 다음과 같이 왼쪽에 `mlflow`라는 이름의 버킷이 생성됩니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232348589-c2e8617b-494f-454f-a829-65a6ba510833.png" title="mlflow-bucket"/>
</p>

---

## Let's Install MLflow Tracking Server

### Helm Repository 추가

```text
helm repo add mlops-for-all https://mlops-for-all.github.io/helm-charts
```

다음과 같은 메시지가 출력되면 정상적으로 추가된 것을 의미합니다.

```text
"mlops-for-all" has been added to your repositories
```

### Helm Repository 업데이트

```text
helm repo update
```

다음과 같은 메시지가 출력되면 정상적으로 업데이트된 것을 의미합니다.

```text
Hang tight while we grab the latest from your chart repositories...
...Successfully got an update from the "mlops-for-all" chart repository
Update Complete. ⎈Happy Helming!⎈
```

### Helm Install

mlflow-server Helm Chart 0.2.0 버전을 설치합니다.

```text
helm install mlflow-server mlops-for-all/mlflow-server \
  --namespace mlflow-system \
  --version 0.2.0
```

- **주의**: 위의 helm chart는 MLflow 의 backend store 와 artifacts store 의 접속 정보를 kubeflow 설치 과정에서 생성한 minio와 위의 [PostgreSQL DB 설치](#postgresql-db-설치)에서 생성한 postgresql 정보를 default로 하여 설치합니다.
  - 별개로 생성한 DB 혹은 Object storage를 활용하고 싶은 경우, [Helm Chart Repo](https://github.com/mlops-for-all/helm-charts/tree/main/mlflow/chart)를 참고하여 helm install 시 value를 따로 설정하여 설치하시기 바랍니다.

다음과 같은 메시지가 출력되어야 합니다.

```text
NAME: mlflow-server
LAST DEPLOYED: Sat Dec 18 22:02:13 2021
NAMESPACE: mlflow-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get pod -n mlflow-system | grep mlflow-server
```

mlflow-system namespace 에 1 개의 mlflow-server 관련 pod 가 Running 이 될 때까지 기다립니다.  
다음과 비슷하게 출력되면 정상적으로 실행된 것입니다.

```text
mlflow-server-ffd66d858-6hm62        1/1     Running   0          74s
```

### 정상 설치 확인

그럼 이제 MLflow Server에 정상적으로 접속되는지 확인해보겠습니다.

우선 클라이언트 노드에서 접속하기 위해, 포트포워딩을 수행합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/mlflow-server-service -n mlflow-system 8080:5000
```

웹 브라우저를 열어 [ServerIP:5995](http://ServerIP:5995)으로 접속하면 다음과 같은 화면이 출력됩니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232348652-46b01b3d-32d2-4919-9f6e-7c6be8fc76c1.png" title="mlflow-install"/>
</p>


## Seldon-Core

Seldon-Core는 쿠버네티스 환경에 수많은 머신러닝 모델을 배포하고 관리할 수 있는 오픈소스 프레임워크 중 하나입니다.  
더 자세한 내용은 Seldon-Core 의 공식 [제품 설명 페이지](https://www.seldon.io/tech/products/core/) 와 [깃헙](https://github.com/SeldonIO/seldon-core) 그리고 API Deployment 파트를 참고해주시기를 바랍니다.

## Selon-Core 설치

Seldon-Core를 사용하기 위해서는 쿠버네티스의 인그레스(Ingress)를 담당하는 Ambassador 와 Istio 와 같은 [모듈이 필요합니다](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html).  
Seldon-Core 에서는 Ambassador 와 Istio 만을 공식적으로 지원하며, *모두의 MLOps*에서는 Ambassador를 사용해 Seldon-core를 사용하므로 Ambassador를 설치하겠습니다.

### Ambassador - Helm Repository 추가

```text
helm repo add datawire https://www.getambassador.io
```

다음과 같은 메시지가 출력되면 정상적으로 추가된 것을 의미합니다.

```text
"datawire" has been added to your repositories
```

### Ambassador - Helm Repository 업데이트

```text
helm repo update
```

다음과 같은 메시지가 출력되면 정상적으로 업데이트된 것을 의미합니다.

```text
Hang tight while we grab the latest from your chart repositories...
...Successfully got an update from the "datawire" chart repository
Update Complete. ⎈Happy Helming!⎈
```

### Ambassador - Helm Install

ambassador Chart 6.9.3 버전을 설치합니다.

```text
helm install ambassador datawire/ambassador \
  --namespace seldon-system \
  --create-namespace \
  --set image.repository=quay.io/datawire/ambassador \
  --set enableAES=false \
  --set crds.keep=false \
  --version 6.9.3
```

다음과 같은 메시지가 출력되어야 합니다.

```text
생략...

W1206 17:01:36.026326   26635 warnings.go:70] rbac.authorization.k8s.io/v1beta1 Role is deprecated in v1.17+, unavailable in v1.22+; use rbac.authorization.k8s.io/v1 Role
W1206 17:01:36.029764   26635 warnings.go:70] rbac.authorization.k8s.io/v1beta1 RoleBinding is deprecated in v1.17+, unavailable in v1.22+; use rbac.authorization.k8s.io/v1 RoleBinding
NAME: ambassador
LAST DEPLOYED: Mon Dec  6 17:01:34 2021
NAMESPACE: seldon-system
STATUS: deployed
REVISION: 1
NOTES:
-------------------------------------------------------------------------------
  Congratulations! You've successfully installed Ambassador!

-------------------------------------------------------------------------------
To get the IP address of Ambassador, run the following commands:
NOTE: It may take a few minutes for the LoadBalancer IP to be available.
     You can watch the status of by running 'kubectl get svc -w  --namespace seldon-system ambassador'

  On GKE/Azure:
  export SERVICE_IP=$(kubectl get svc --namespace seldon-system ambassador -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

  On AWS:
  export SERVICE_IP=$(kubectl get svc --namespace seldon-system ambassador -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

  echo http://$SERVICE_IP:

For help, visit our Slack at http://a8r.io/Slack or view the documentation online at https://www.getambassador.io.
```

seldon-system 에 4 개의 pod 가 Running 이 될 때까지 기다립니다.

```text
kubectl get pod -n seldon-system
```

```text
ambassador-7f596c8b57-4s9xh                  1/1     Running   0          7m15s
ambassador-7f596c8b57-dt6lr                  1/1     Running   0          7m15s
ambassador-7f596c8b57-h5l6f                  1/1     Running   0          7m15s
ambassador-agent-77bccdfcd5-d5jxj            1/1     Running   0          7m15s
```

### Seldon-Core - Helm Install

seldon-core-operator Chart 1.11.2 버전을 설치합니다.

```text
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --namespace seldon-system \
    --set usageMetrics.enabled=true \
    --set ambassador.enabled=true \
    --version 1.11.2
```

다음과 같은 메시지가 출력되어야 합니다.

```text
생략...

W1206 17:05:38.336391   28181 warnings.go:70] admissionregistration.k8s.io/v1beta1 ValidatingWebhookConfiguration is deprecated in v1.16+, unavailable in v1.22+; use admissionregistration.k8s.io/v1 ValidatingWebhookConfiguration
NAME: seldon-core
LAST DEPLOYED: Mon Dec  6 17:05:34 2021
NAMESPACE: seldon-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

seldon-system namespace 에 1 개의 seldon-controller-manager pod 가 Running 이 될 때까지 기다립니다.

```text
kubectl get pod -n seldon-system | grep seldon-controller
```

```text
seldon-controller-manager-8457b8b5c7-r2frm   1/1     Running   0          2m22s
```

## References

- [Example Model Servers with Seldon](https://docs.seldon.io/projects/seldon-core/en/latest/examples/server_examples.html#examples-server-examples--page-root)



## Prometheus & Grafana

프로메테우스(Prometheus) 와 그라파나(Grafana) 는 모니터링을 위한 도구입니다.  
안정적인 서비스 운영을 위해서는 서비스와 서비스가 운영되고 있는 인프라의 상태를 지속해서 관찰하고, 관찰한 메트릭을 바탕으로 문제가 생길 때 빠르게 대응해야 합니다.  
이러한 모니터링을 효율적으로 수행하기 위한 많은 도구 중 *모두의 MLOps*에서는 오픈소스인 프로메테우스와 그라파나를 사용할 예정입니다.

더 자세한 내용은 [Prometheus 공식 문서](https://prometheus.io/docs/introduction/overview/), [Grafana 공식 문서](https://grafana.com/docs/)를 확인해주시기를 바랍니다.

프로메테우스는 다양한 대상으로부터 Metric을 수집하는 도구이며, 그라파나는 모인 데이터를 시각화하는 것을 도와주는 도구입니다. 서로 간의 종속성은 없지만 상호 보완적으로 사용할 수 있어 함께 사용되는 경우가 많습니다.

이번 페이지에서는 쿠버네티스 클러스터에 프로메테우스와 그라파나를 설치한 뒤, Seldon-Core 로 생성한 SeldonDeployment 로 API 요청을 보내, 정상적으로 Metrics 이 수집되는지 확인해보겠습니다.

본 글에서는 seldonio/seldon-core-analytics Helm Chart 1.12.0 버전을 활용해 쿠버네티스 클러스터에 프로메테우스와 그라파나를 설치하고, Seldon-Core 에서 생성한 SeldonDeployment의 Metrics 을 효율적으로 확인하기 위한 대시보드도 함께 설치합니다.

### Helm Repository 추가

```text
helm repo add seldonio https://storage.googleapis.com/seldon-charts
```

다음과 같은 메시지가 출력되면 정상적으로 추가된 것을 의미합니다.

```text
"seldonio" has been added to your repositories
```

### Helm Repository 업데이트

```text
helm repo update
```

다음과 같은 메시지가 출력되면 정상적으로 업데이트된 것을 의미합니다.

```text
Hang tight while we grab the latest from your chart repositories...
...Successfully got an update from the "seldonio" chart repository
...Successfully got an update from the "datawire" chart repository
Update Complete. ⎈Happy Helming!⎈
```

### Helm Install

seldon-core-analytics Helm Chart 1.12.0 버전을 설치합니다.

```text
helm install seldon-core-analytics seldonio/seldon-core-analytics \
  --namespace seldon-system \
  --version 1.12.0
```

다음과 같은 메시지가 출력되어야 합니다.

```text
생략...
NAME: seldon-core-analytics
LAST DEPLOYED: Tue Dec 14 18:29:38 2021
NAMESPACE: seldon-system
STATUS: deployed
REVISION: 1
```

정상적으로 설치되었는지 확인합니다.

```text
kubectl get pod -n seldon-system | grep seldon-core-analytics
```

seldon-system namespace 에 6개의 seldon-core-analytics 관련 pod 가 Running 이 될 때까지 기다립니다.

```text
seldon-core-analytics-grafana-657c956c88-ng8wn                  2/2     Running   0          114s
seldon-core-analytics-kube-state-metrics-94bb6cb9-svs82         1/1     Running   0          114s
seldon-core-analytics-prometheus-alertmanager-64cf7b8f5-nxbl8   2/2     Running   0          114s
seldon-core-analytics-prometheus-node-exporter-5rrj5            1/1     Running   0          114s
seldon-core-analytics-prometheus-pushgateway-8476474cff-sr4n6   1/1     Running   0          114s
seldon-core-analytics-prometheus-seldon-685c664894-7cr45        2/2     Running   0          114s
```

### 정상 설치 확인

그럼 이제 그라파나에 정상적으로 접속되는지 확인해보겠습니다.

우선 클라이언트 노드에서 접속하기 위해, 포트포워딩을 수행합니다.

```text
kubectl port-forward --address 0.0.0.0 svc/seldon-core-analytics-grafana -n seldon-system 8080:80
```

웹 브라우저를 열어 [ServerIP:5995](http://ServerIP:5995)으로 접속하면 다음과 같은 화면이 출력됩니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232349069-5e346127-8e84-478e-87bd-2c8310620070.png" title="grafana-install"/>
</p>

다음과 같은 접속정보를 입력하여 접속합니다.

- Email or username : `admin`
- Password : `password`


좌측의 대시보드 아이콘을 클릭하여, `Manage` 버튼을 클릭합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232349087-0dffa700-6aa9-47ee-a6cd-989971cb1cb2.png" title="grafana-install"/>
</p>

기본적인 그라파나 대시보드가 포함되어있는 것을 확인할 수 있습니다. 이 중 `Prediction Analytics` 대시보드를 클릭합니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232349093-7bffce24-a8c3-48b8-84ff-70f07eb3baea.png" title="grafana-install"/>
</p>

Seldon Core API Dashboard 가 보이고, 다음과 같이 출력되는 것을 확인할 수 있습니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/40382596/232349101-aace5deb-59d4-4809-9970-766f7c10040e.png" title="grafana"/>
</p>

## References

- [Seldon-Core-Analytics Helm Chart](https://github.com/SeldonIO/seldon-core/tree/master/helm-charts/seldon-core-analytics)


## Kubeflow에서 Notebook 생성 시 Could not find CSRF cookie XSRF-TOKEN in the request 에러 발생할 때
출처 : https://otzslayer.github.io/kubeflow/2022/06/11/could-not-find-csrf-cookie-xsrf-token-in-the-request.html

Kubeflow를 설치하고 가장 먼저 접하는 컴포넌트는 아무래도 Notebook 환경이 됩니다. </BR>
적어도 “Hello, world!” 같은 느낌으로 시작을 해보려면 Notebook 생성을 한 번 정도는 하게 되는데요. </BR>
시작과 동시에 다음과 같은 에러를 접하게 되는 경우가 있습니다.
```
[403] Could not find CSRF cookie XSRF-TOKEN in the request. 
http://<IP Address>:8080/jupyter/api/namespaces/kubeflow-user-example-com/notebooks
```
XSRF-TOKEN 이야기가 나오는걸로 봐선 보안 관련 문제인 것 같아 겨우 해결 방법을 찾게 되었습니다.

### 어떤 문제?
### 문제 분석
제가 사용하고 있는 환경은 사실 로컬호스트가 아닌 원격 환경이었습니다. 우선 WSL에 Kubeflow를 띄워 포트포워딩하여 사용하고 있었고, </BR>
WSL 아이피를 로컬호스트로 할당한 상태에서 외부에서 접근할 수 있게 설정해 맥북에서 아이피와 포트를 통해 접근하고 있었습니다. </BR>
그래서 Notebook 생성이 어떤 경우에도 안되는 것인가 싶어 로컬호스트 주소인 localhost:8080로 들어간 다음 생성해보니 문제 없이 생성되는 것을 확인했습니다. </BR>
처음에는 원격에서는 원래 생성이 안되는 것인가 싶었죠. 조금 시간을 들여 검색해보니 HTTP로 접근하게 되면 발생하는 오류인 것을 알게 되었습니다.</BR>
</BR>
이게 무슨 상황인가 싶어서 보니 로컬호스트는 HTTPS로 대시보드를 접근하게 되어 있었고 아이피를 통한 접근은 모두 HTTP로 접근하게 되어 있었던거죠. </BR>
원격으로 접근하는 것 역시 아이피를 통한 접근이었기 때문에 생성이 되지 않았던 것입니다.</BR>

### 문제 해결
제가 찾은 문제를 해결할 수 있는 방법은 두 가지입니다. </BR>
하나는 HTTP 접근을 허용하는 것, 나머지 하나는 self-signing-issuer를 이용하는 것인데요. </BR>
물론 보안 측면에서는 후자를 택하는 것이 일반적이고 권장됩니다. 하지만 해결하는 방법이 생각보다 쉽지는 않았습니다. </BR>
실제로 시도해 보았지만 오히려 HTTPS 접근할 때 접속이 아예 되지 않아 Kubeflow를 재설치하는 삽질을 하고 말았구요. </BR>
제가 실수한 부분이 클텐데 권장하는 해결 방안은 나중에 시도해보기로 하고 당장 가볍게 개인 용도로 사용하기 위해 간단한 방법으로 해결하였습니다. </BR>
올바른 해결 방법은 [https://github.com/mlops-for-all/mlops-for-all.github.io/issues/72#issuecomment-1007537301](여기)를 참고하시기 바랍니다. 자세하게 설명되어 있습니다.</BR>
</BR>
Notebook 생성 시 HTTP 허용을 위해서는 Jupyter의 재설치가 필요합니다. </BR>
재설치에 앞서 Jupyter 설정에 Secure Cookie를 해제하는 설정값을 추가해야 합니다. </BR>
설치 파일이 있는 폴더에서 다음 파일을 수정하면 됩니다.</BR>
```
manifest/apps/jupyter/jupyter-web-app/upstream/base/deployment.yaml
```
파일을 열어 다음의 내용을 추가합니다.
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment
spec:
  replicas: 1
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: "false"
    spec:
      containers:
      - name: jupyter-web-app
        image: public.ecr.aws/j1r0q0g6/notebooks/jupyter-web-app
        ports:
        - containerPort: 5000
        volumeMounts:
        - mountPath: /etc/config
          name: config-volume
        - mountPath: /src/apps/default/static/assets/logos
          name: logos-volume
        env:
        - name: APP_PREFIX
          value: $(JWA_PREFIX)
        - name: UI
          value: $(JWA_UI)
        - name: USERID_HEADER
          value: $(JWA_USERID_HEADER)
        - name: USERID_PREFIX
          value: $(JWA_USERID_PREFIX)
        # 여기서부터 추가합니다.
        - name: APP_SECURE_COOKIES
          value: "false"
        # 위 내용까지 추가합니다.
      serviceAccountName: service-account
      volumes:
      - configMap:
          name: config
        name: config-volume
      - configMap:
          name: jupyter-web-app-logos
        name: logos-volume
```
그 다음 다음 명령어를 통해 다시 설치하시면 됩니다. 물론 아래 명령어가 아니라 Kubeflow 원클릭 설치 방법으로도 가능합니다. </BR>
어차피 변경 사항이 없는 Pod들은 영향을 받지 않으니까요.</BR>
```
kustomize build apps/jupyter/jupyter-web-app/upstream/overlay/istio | kubectl apply -f -
```
이후 다시 포트포워딩하여 대시보드에 접속해 Notebook을 생성해보면 정상적으로 생성되는 것을 확인하실 수 있습니다.