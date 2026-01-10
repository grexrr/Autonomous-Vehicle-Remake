pipeline {
    agent any

    environment {
        AWS_DEFAULT_REGION = 'us-east-1'
        INSTANCE_ID = 'i-02b51d076ac8a13a1'
        AWS_CRED_ID = 'grexrr_cicd'
        DOCKERHUB_CRED_ID = 'dockerhub'
        IMAGE_REPO = 'grexrr/autonomous-vehicle'
    }

    stages {
        stage('Checkout'){
            steps {
                checkout scm
                sh 'ls -la'
                sh 'test -f Dockerfile'
            }
        }

        // stage('Build Image') {
        //     steps {
        //         withCredentials([usernamePassword(
        //             credentialsId: "${env.DOCKERHUB_CRED_ID}",
        //             usernameVariable: 'DOCKERHUB_USER',
        //             passwordVariable: 'DOCKERHUB_TOKEN'
        //         )]){
        //             script {
        //                 // 1. login docker
        //                 sh '''
        //                     if ! grep -q "docker.io" ~/.docker/config.json 2>/dev/null; then
        //                         echo "Dockerhub not logged in, logging in now..."
        //                         echo $DOCKERHUB_TOKEN | docker login -u $DOCKERHUB_USER --password-stdin
        //                     else
        //                         echo "Already logged in to Docker Hub"
        //                     fi
        //                 '''

        //                 // 2. build + push 
        //                 sh '''
        //                     set -e
        //                     docker buildx create --use --name 
        //                 '''
                        
        //             }
        //         }
        //     }
        // }

        // stage('Pull & Compose Autonomous Vehicle Service') {
        //     steps {
        //         withCredentials([[
        //             $class: 'AmazonWebServicesCredentialsBinding',
        //             credentialsId: "${env.AWS_CRED_ID}"
        //         ]]) {
        //             script {
        //                 // 1. acquire
        //                 sh '''
        //                     echo "Inside EC2"
        //                 '''
        //             }
        //         }
        //     }
        // }
    }
}
