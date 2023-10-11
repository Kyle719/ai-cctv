# Task 2 - Loadbalancing on ingess

### Ingress 의 구현체인 Contorller 설치와 로드밸런싱 실습
#

1. 웹브라우저로 아래 링크에 접속하여 ingress controller 설치 전 확인
```
https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/baremetal/deploy.yaml
```

2. nginx ingress controller 설치
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/baremetal/deploy.yaml
```  

3. 홈페이지에 관한 Service, Pod yaml 확인
```
cat homepage.yaml
```

4. 위 yaml 로 리소스 생성
```
kubectl create -f homepage.yaml
```

5. 고객센터에 관한 Service, Pod yaml 확인
```
cat customer.yaml
```

6. 위 yaml 로 리소스 생성
```
kubectl create -f customer.yaml
```

7. 연간 스케쥴 페이지에 관한 Service, Pod yaml 확인
```
cat schedule.yaml
```

8. 위 yaml 로 리소스 생성
```
kubectl create -f schedule.yaml
```

9. 로드밸런싱 용 ingress yaml 확인
```
cat lb-ingress.yaml
```

10. 위 yaml 로 리소스 생성
```
kubectl create -f lb-ingress.yaml
```

11. 위에서 생성한 리소스 확인
```
kubectl get po
kubectl get service
kubectl get ingress
```

12. ingress controller 가 만든 Nodeport 유형의 서비스의 Nodeport를 확인
```
kubectl get svc -n ingress-nginx
```

13. 워커노드의 ip와 12과정에서 확인한 Nodeport를 확인하여 접근시도
```
curl <worker ip>:<12에서 확인한 portnumber>
curl <worker ip>:<12에서 확인한 portnumber>/customer
curl <worker ip>:<12에서 확인한 portnumber>/schedule
```

14. clear
```
kubectl delete pod,svc,ingress --all
```
