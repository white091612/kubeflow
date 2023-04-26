# kubeflow 설치 가이드
## 1. minikube 셋업
### docker 컨테이너 접속

```
docker start joseph.kang_kubeflow
docker attach joseph.kang_kubeflow
```
### add GPG key
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
### PID1 관련 에러가 무조건 뜨니까 아래 줄 입력
```
sudo apt-get update && sudo apt-get install -yqq daemonize dbus-user-session fontconfig
sudo daemonize /usr/bin/unshare --fork --pid --mount-proc /lib/systemd/systemd --system-unit=basic.target
nano /etc/sudoers
exec sudo nsenter -t $(pidof systemd) -a su - $LOGNAME
```
### root 계정으로는 minikkube 설치가 안되기 때문에 user를 추가합니다.
```
usermod –aG sudo joseph
su - joseph
sudo usermod -aG docker $USER && newgrp docker
```
### 