# Task2 - Configmap, Secret

### 환경설정 정보를 관리하는 Configmap 과 Secret 사용법
#  

1. yaml 확인

```
cat configmap.yaml secret.yaml
```  

2. yaml 을 활용한 configmap 생성
```
kubectl create -f configmap.yaml
kubectl create -f secret.yaml
```

3. 생성 확인
```
kubectl get configmap
kubectl get secret
```

4. 실습에 사용할 pod yaml 확인
```
cat cs-pod.yaml
```

5. 위 4에서 확인한 pod 생성
```
kubectl create -f cs-pod.yaml
```

6. 위 5에서 생성한 pod의 컨테이너로 접속
```
kubectl exec -it cs-pod --container container -- /bin/bash
```

7. configmap 과 secret이 추가한 환경설정 확인
```
env
exit
```

8. 간단한 내용으로 txt 파일 생성
```
echo "cm-content" >> cfile.txt
echo "sc-content" >> sfile.txt
```

9. 위 8 에서 만든 파일을 활용한 configmap 과 secret을 생성
```
kubectl create configmap fcm --from-file=./cfile.txt
kubectl create secret generic fsc --from-file=./sfile.txt
```

10. file 기반 환경설정 실습에 사용할 Pod의 yaml 확인
```
cat filecs-pod.yaml
```

11. 위 10에서 확인한 yaml로 pod생성
```
kubectl create -f filecs-pod.yaml
```

12. 위 11에서 생성한 Pod의 컨테이너로 접속
```
kubectl exec -it filecs-pod --container container -- /bin/bash
```

13. file 기반 Configmap, Secret이 추가한 환경설정 확인
```
env
exit
```

14. clear
```
kubectl delete po,configmap,secret --all
```
