# Task 3 - Canary Upgrade

### 애플리케이션의 업데이트 방식중 하나인 카나리 업그레이드 기능 실습
#

1. v1 의 yaml 파일 확인
```
cat v1.yaml
```

2. 위 yaml 로 리소스 생성
```
kubectl create -f v1.yaml
```

3. v1에 대한 ingress yaml 확인
```
cat canary_v1-ingress.yaml
```

4. 위 yaml 로 리소스 생성
```
kubectl create -f canary_v1-ingress.yaml
```

5. hosts 파일 수정
```
echo <masterIP> www.canary.com >> /etc/hosts
cat /etc/hosts
```

6. ingress-nginx-controller 서비스를 확인
```
kubectl get svc -n ingress-nginx
```

7. 해당 노드포트를 활용하여 curl 수행
```
curl www.canary.com:<위 결과에서 확인한 포트>
```

8. v2 의 yaml 확인
```
cat v2.yaml
```

9. 위 yaml 로 리소스 생성
```
kubectl create -f v2.yaml
```

10. v2에 대한 ingress yaml 확인
```
cat canary_v2-ingress.yaml
```

11. 위 yaml 로 리소스 생성
```
kubectl create -f canary_v2-ingress.yaml
```

12. 해당 명령어로 curl 반복 수행
```
while true; do curl www.canary.com:<위 7번과정에서 입력했던 포트>; sleep 1; done
```
ctrl + c 키를 입력하여 취소

13. 리소스 삭제
```
kubectl delete pod,svc,ingress --all
```
