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
                sh 'echo "Successful!"'
            }
        }

        stage('Generate Tag'){
            steps {
                script {
                    env.IMAGE_TAG = sh(
                        returnStdout: true,
                        script: "date -u +%Y%m%d-%H%M%S"
                    ).trim()
                    echo "IMAGE_TAG=${env.IMAGE_TAG}"
                }
            }
        }

        stage('Build & Push Image') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: "${env.DOCKERHUB_CRED_ID}",
                    usernameVariable: 'DOCKERHUB_USER',
                    passwordVariable: 'DOCKERHUB_TOKEN'
                )]){
                    sh '''
                        set -e

                        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin

                        docker buildx create --use --name jxbuilder || docker buildx use jxbuilder

                        docker buildx inspect --bootstrap

                        docker buildx build \
                        --platform linux/arm64 \
                        -t "${IMAGE_REPO}:${IMAGE_TAG}" \
                        --push .
                    '''
                }
            }
        }

        stage('Deploy on EC2 via SSM') {
            steps {
                withCredentials([[
                    $class: 'AmazonWebServicesCredentialsBinding',
                    credentialsId: "${env.AWS_CRED_ID}"
                ]]){
                    script {
                        try {
                            def region = env.AWS_DEFAULT_REGION
                            def instanceId = env.INSTANCE_ID
                            def imageTag = env.IMAGE_TAG
                            
                            def cmdId = sh(
                                returnStdout: true,
                                script: """
                                    aws ssm send-command \\
                                        --region '${region}' \\
                                        --document-name "AWS-RunShellScript" \\
                                        --instance-ids '${instanceId}' \\
                                        --parameters 'commands=[
                                            "set -e",
                                            "cd /home/ubuntu/autonomous-vehicle",
                                            "test -f .env || touch .env",
                                            "OLD_TAG=\\$(grep -E \\"^IMAGE_TAG=\\" .env | tail -n 1 | cut -d= -f2- || true)",
                                            "if [ -n \\"\\\\$OLD_TAG\\" ]; then if grep -qE \\"^PREV_IMAGE_TAG=\\" .env; then sed -i \\"s/^PREV_IMAGE_TAG=.*/PREV_IMAGE_TAG=\\\\$OLD_TAG/\\" .env; else echo \\"PREV_IMAGE_TAG=\\\\$OLD_TAG\\" >> .env; fi; fi",
                                            "if grep -qE \\"^IMAGE_TAG=\\" .env; then sed -i \\"s/^IMAGE_TAG=.*/IMAGE_TAG=${env.IMAGE_TAG}/\\" .env; else echo \\"IMAGE_TAG=${env.IMAGE_TAG}\\" >> .env; fi",
                                            "cat .env | egrep \\"^(IMAGE_TAG|PREV_IMAGE_TAG)=\\" || true",
                                            "docker compose config | grep image",
                                            "docker compose pull",
                                            "docker compose up -d --remove-orphans",
                                            "curl -fsS http://localhost:5000/api/vehicle/health",
                                            "docker image prune -af --filter \\"until=168h\\""
                                        ]' \\
                                        --query "Command.CommandId" --output text
                                """
                            ).trim()

                            echo "SSM CommandId: ${cmdId}"

                            sh """
                                aws ssm wait command-executed \\
                                    --region '${region}' \\
                                    --command-id '${cmdId}' \\
                                    --instance-id '${instanceId}'
                            """

                            def status = sh(
                                returnStdout: true,
                                script: """
                                    aws ssm get-command-invocation \\
                                        --region '${region}' \\
                                        --command-id '${cmdId}' \\
                                        --instance-id '${instanceId}' \\
                                        --query "Status" --output text
                                """
                            ).trim()

                            def out = sh(
                                returnStdout: true, 
                                script: """
                                    aws ssm get-command-invocation \\
                                        --region '${region}' \\
                                        --command-id '${cmdId}' \\
                                        --instance-id '${instanceId}' \\
                                        --query "StandardOutputContent" --output text
                                """
                            ).trim()

                            def err = sh(
                                returnStdout: true, 
                                script: """
                                    aws ssm get-command-invocation \\
                                        --region '${region}' \\
                                        --command-id '${cmdId}' \\
                                        --instance-id '${instanceId}' \\
                                        --query "StandardErrorContent" --output text
                                """
                            ).trim()

                            echo "SSM Status: ${status}"
                            echo "STDOUT:\\n${out}"
                            if (err && err != 'None') echo "STDERR:\\n${err}"
                            if (status != 'Success') error("Deploy failed: ${status}")
                        } catch (e) {
                            echo "Deploy failed, starting rollback... Reason: ${e}"
                            
                            def region = env.AWS_DEFAULT_REGION
                            def instanceId = env.INSTANCE_ID
                            
                            def rbCmdId = sh(
                                returnStdout: true,
                                script: """
                                    aws ssm send-command \\
                                        --region '${region}' \\
                                        --document-name "AWS-RunShellScript" \\
                                        --instance-ids '${instanceId}' \\
                                        --parameters 'commands=[
                                            "set -e",
                                            "cd /home/ubuntu/autonomous-vehicle",
                                            "test -f .env || (echo \\".env missing\\" && exit 2)",
                                            "PREV=\\$(grep -E \\"^PREV_IMAGE_TAG=\\" .env | tail -n 1 | cut -d= -f2- || true)",
                                            "if [ -z \\"\\\\$PREV\\" ]; then echo \\"No PREV_IMAGE_TAG found, cannot rollback\\"; exit 3; fi",
                                            "if grep -qE \\"^IMAGE_TAG=\\" .env; then sed -i \\"s/^IMAGE_TAG=.*/IMAGE_TAG=\\\\$PREV/\\" .env; else echo \\"IMAGE_TAG=\\\\$PREV\\" >> .env; fi",
                                            "cat .env | egrep \\"^(IMAGE_TAG|PREV_IMAGE_TAG)=\\" || true",
                                            "docker compose pull",
                                            "docker compose up -d --remove-orphans",
                                            "curl -fsS http://localhost:5000/api/vehicle/health"
                                            ]' \\
                                        --query "Command.CommandId" --output text
                                    """
                                ).trim()

                            echo "SSM CommandId (rollback): ${rbCmdId}"

                            sh """
                                aws ssm wait command-executed \\
                                    --region '${region}' \\
                                    --command-id '${rbCmdId}' \\
                                    --instance-id '${instanceId}'
                            """

                            def rbStatus = sh(
                                returnStdout: true, 
                                script: """
                                    aws ssm get-command-invocation \\
                                        --region '${region}' \\
                                        --command-id '${rbCmdId}' \\
                                        --instance-id '${instanceId}' \\
                                        --query "Status" --output text
                                """
                            ).trim()

                            def rbOut = sh(
                                returnStdout: true, 
                                script: """
                                    aws ssm get-command-invocation \\
                                        --region '${region}' \\
                                        --command-id '${rbCmdId}' \\
                                        --instance-id '${instanceId}' \\
                                        --query "StandardOutputContent" --output text
                                """
                            ).trim()

                            def rbErr = sh(
                                returnStdout: true, 
                                script: """
                                    aws ssm get-command-invocation \\
                                        --region '${region}' \\
                                        --command-id '${rbCmdId}' \\
                                        --instance-id '${instanceId}' \\
                                        --query "StandardErrorContent" --output text
                                """
                            ).trim()

                            echo "SSM Status (rollback): ${rbStatus}"
                            echo "STDOUT (rollback):\n${rbOut}"
    
                            if (rbErr && rbErr != 'None') echo "STDERR (rollback):\n${rbErr}"
                            if (rbStatus != 'Success') {
                                error("Deploy failed AND rollback failed: ${rbStatus}")
                            }
                            error("Deploy failed, rollback succeeded. Build marked as FAILED on purpose.")
                        }
                    }
                }
            }
        }
    }
}
