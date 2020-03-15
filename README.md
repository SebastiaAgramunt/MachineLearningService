# Machine learning service

This project aims to reproduce a machine learning service on production. It consists on all the parts needed, a download-preprocessing phase to prepare the data, a training phase to train a machine learning model and a Rest API service that allows an external client to perform queries to get the predictions.

### Prerequisites

Make sure you have installed [Docker](https://www.docker.com/get-started) in your computer. Try to get your docker version on the command line

```sh 
docker --version
```
It's been tested on  ```Docker version 19.03.5``` but upper versions may work as well.

### Running

In the main folder run

```sh
make download-files
make train
make start-server
```

where the first command will download a public dataset and transform the data for proper training. The second command will train the model and the third one will start the API service (by default on [http://127.0.0.1:5000/](http://127.0.0.1:5000/))

To fake a client we can run on another terminal (having started the server):

```sh
make start-client
```

If you like the project, please star it!.