
Simple example of NN (Perceptron) to run in Flask WEB server as Docker image.
Given the following X predict y:

<pre>
  X   | y
---------
0 0 1 | 0
1 1 0 | 1
1 0 1 | 1
0 1 1 | 0
----------
1 0 0 | ?
</pre>

TRAIN MODEL FIRST BEFORE PREDICTING

* To run from this directory:

python nn.py

* To train NN:

http://127.0.0.1:5078/train

Train model and get predictions on the traing set

Learning rate must be in range: 0.01 <= lr <= 0.02
Enter learning rate: 0.01

* To predict new data:

http://127.0.0.1:5078/predict

Predict classes for new data

Input format: [[x11,x12,x13],[x21,x22,x23], ...[xn1,xn2,xn3]], 
where x = {0,1}, for example [[1,0,0], [1,1,1], ...]

* To build and run Docker image:

docker build --tag nn-app .
docker run  -p 5078:5078 nn-app

* To put in repository:

docker tag nn-app dokondr/nn-app
docker push dokondr/nn-app

* Repository:

https://hub.docker.com/r/dokondr/nn-app

To delete all containers including its volumes use,
docker rm -vf $(docker ps -a -q)

To delete all the images,
docker rmi -f $(docker images -a -q)
