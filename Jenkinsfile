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

                        docker buildx inspect jxbuilder >/dev/null 2>&1 || docker buildx create --use --name jxbuilder
                        docker buildx use jxbuilder

                        docker buildx build \
                        --platform linux/arm64 \
                        -t "${IMAGE_REPO}:${IMAGE_TAG}" \
                        --push .
                    '''
                }
            }
        }

        stage('AWS Identity (sanity)') {
            steps {
                withCredentials([[
                    $class: 'AmazonWebServicesCredentialsBinding',
                    credentialsId: "${env.AWS_CRED_ID}"
                ]]) {
                    sh '''
                        set -e
                        aws --version
                        aws sts get-caller-identity --region "$AWS_DEFAULT_REGION"
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
                        // 设置环境变量，供 shell 脚本使用
                        env.DEPLOY_REGION = env.AWS_DEFAULT_REGION
                        env.DEPLOY_INSTANCE_ID = env.INSTANCE_ID
                        env.DEPLOY_IMAGE_TAG = env.IMAGE_TAG
                        
                        try {
                            // 使用 --cli-input-json 文件方式，避免 Groovy 字符串插值问题
                            def deployCommands = [
                                'set -e',
                                'cd /home/ubuntu/autonomous-vehicle',
                                'test -f .env || touch .env',
                                'OLD_TAG=$(grep -E "^IMAGE_TAG=" .env | tail -n 1 | cut -d= -f2- || true)',
                                'if [ -n "$OLD_TAG" ]; then if grep -qE "^PREV_IMAGE_TAG=" .env; then sed -i "s/^PREV_IMAGE_TAG=.*/PREV_IMAGE_TAG=$OLD_TAG/" .env; else echo "PREV_IMAGE_TAG=$OLD_TAG" >> .env; fi; fi',
                                "if grep -qE \"^IMAGE_TAG=\" .env; then sed -i \"s/^IMAGE_TAG=.*/IMAGE_TAG=${env.DEPLOY_IMAGE_TAG}/\" .env; else echo \"IMAGE_TAG=${env.DEPLOY_IMAGE_TAG}\" >> .env; fi",
                                'docker compose config | grep image',
                                'docker compose pull',
                                'docker compose up -d --remove-orphans',
                                'sleep 3',
                                'HEALTH_OK=0; for i in $(seq 1 15); do if curl -fsS http://localhost:5000/api/vehicle/health; then HEALTH_OK=1; break; fi; echo "health check retry $i"; sleep 2; done; if [ $HEALTH_OK -ne 1 ]; then echo "health check failed, dumping logs"; docker compose logs --no-color --tail 200 api || true; exit 1; fi',
                                'docker image prune -af --filter "until=168h"'
                            ]

                            def deployPayload = [
                                DocumentName: 'AWS-RunShellScript',
                                InstanceIds: [env.DEPLOY_INSTANCE_ID],
                                Parameters: [commands: deployCommands]
                            ]

                            writeFile(
                                file: '/tmp/ssm-deploy.json',
                                text: groovy.json.JsonOutput.prettyPrint(
                                    groovy.json.JsonOutput.toJson(deployPayload)
                                )
                            )

                            sh '''
                                set -e
                                echo "=== ssm-deploy.json ==="
                                cat /tmp/ssm-deploy.json
                                echo "======================="
                            '''.stripIndent()

                            def cmdId = sh(
                                returnStdout: true,
                                script: '''
                                    set -e
                                    CMD_ID=$(aws ssm send-command --region "$DEPLOY_REGION" --cli-input-json file:///tmp/ssm-deploy.json --query 'Command.CommandId' --output text)
                                    echo "$CMD_ID" | grep -Eq '^[0-9a-f-]{36}$' || (echo "Invalid CommandId format: $CMD_ID" >&2 && exit 1)
                                    printf '%s' "$CMD_ID"
                                '''.stripIndent()
                            ).trim()

                            sh '''
                                set -e
                                rm -f /tmp/ssm-deploy.json
                            '''.stripIndent()

                            echo "SSM CommandId: ${cmdId}"
                            
                            env.DEPLOY_CMD_ID = cmdId

                            def waitRc = sh(
                                returnStatus: true,
                                script: '''
                                    aws ssm wait command-executed \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$DEPLOY_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID"
                                '''.stripIndent()
                            )
                            if (waitRc != 0) {
                                echo "SSM wait failed with rc=${waitRc}, fetching invocation output..."
                            }

                            def status = sh(
                                returnStdout: true,
                                script: '''
                                    aws ssm get-command-invocation \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$DEPLOY_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID" \
                                        --query "Status" --output text
                                '''
                            ).trim()

                            def out = sh(
                                returnStdout: true, 
                                script: '''
                                    aws ssm get-command-invocation \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$DEPLOY_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID" \
                                        --query "StandardOutputContent" --output text
                                '''
                            ).trim()

                            def err = sh(
                                returnStdout: true, 
                                script: '''
                                    aws ssm get-command-invocation \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$DEPLOY_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID" \
                                        --query "StandardErrorContent" --output text
                                '''
                            ).trim()

                            echo "SSM Status: ${status}"
                            echo "STDOUT:\n${out}"
                            if (err && err != 'None') echo "STDERR:\n${err}"
                            if (status != 'Success') error("Deploy failed: ${status}")
                        } catch (e) {
                            echo "Deploy failed, starting rollback... Reason: ${e}"
                            
                            // 回滚也使用 --cli-input-json 方式
                            def rollbackCommands = [
                                'set -e',
                                'cd /home/ubuntu/autonomous-vehicle',
                                'test -f .env || (echo ".env missing" && exit 2)',
                                'PREV=$(grep -E "^PREV_IMAGE_TAG=" .env | tail -n 1 | cut -d= -f2- || true)',
                                'if [ -z "$PREV" ]; then echo "No PREV_IMAGE_TAG found, cannot rollback"; exit 3; fi',
                                'if grep -qE "^IMAGE_TAG=" .env; then sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$PREV/" .env; else echo "IMAGE_TAG=$PREV" >> .env; fi',
                                'docker compose pull',
                                'docker compose up -d --remove-orphans',
                                'sleep 3',
                                'HEALTH_OK=0; for i in $(seq 1 15); do if curl -fsS http://localhost:5000/api/vehicle/health; then HEALTH_OK=1; break; fi; echo "health check retry $i"; sleep 2; done; if [ $HEALTH_OK -ne 1 ]; then echo "health check failed, dumping logs"; docker compose logs --no-color --tail 200 api || true; exit 1; fi'
                            ]

                            def rollbackPayload = [
                                DocumentName: 'AWS-RunShellScript',
                                InstanceIds: [env.DEPLOY_INSTANCE_ID],
                                Parameters: [commands: rollbackCommands]
                            ]

                            writeFile(
                                file: '/tmp/ssm-rollback.json',
                                text: groovy.json.JsonOutput.prettyPrint(
                                    groovy.json.JsonOutput.toJson(rollbackPayload)
                                )
                            )

                            sh '''
                                set -e
                                echo "=== ssm-rollback.json ==="
                                cat /tmp/ssm-rollback.json
                                echo "======================="
                            '''.stripIndent()

                            def rbCmdId = sh(
                                returnStdout: true,
                                script: '''
                                    set -e
                                    CMD_ID=$(aws ssm send-command --region "$DEPLOY_REGION" --cli-input-json file:///tmp/ssm-rollback.json --query 'Command.CommandId' --output text)
                                    echo "$CMD_ID" | grep -Eq '^[0-9a-f-]{36}$' || (echo "Invalid CommandId format: $CMD_ID" >&2 && exit 1)
                                    printf '%s' "$CMD_ID"
                                '''.stripIndent()
                            ).trim()

                            sh '''
                                set -e
                                rm -f /tmp/ssm-rollback.json
                            '''.stripIndent()

                            echo "SSM CommandId (rollback): ${rbCmdId}"
                            
                            env.ROLLBACK_CMD_ID = rbCmdId

                            def rbWaitRc = sh(
                                returnStatus: true,
                                script: '''
                                    aws ssm wait command-executed \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$ROLLBACK_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID"
                                '''.stripIndent()
                            )
                            if (rbWaitRc != 0) {
                                echo "SSM wait failed (rollback) rc=${rbWaitRc}, fetching invocation output..."
                            }

                            def rbStatus = sh(
                                returnStdout: true, 
                                script: '''
                                    aws ssm get-command-invocation \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$ROLLBACK_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID" \
                                        --query "Status" --output text
                                '''
                            ).trim()

                            def rbOut = sh(
                                returnStdout: true, 
                                script: '''
                                    aws ssm get-command-invocation \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$ROLLBACK_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID" \
                                        --query "StandardOutputContent" --output text
                                '''
                            ).trim()

                            def rbErr = sh(
                                returnStdout: true, 
                                script: '''
                                    aws ssm get-command-invocation \
                                        --region "$DEPLOY_REGION" \
                                        --command-id "$ROLLBACK_CMD_ID" \
                                        --instance-id "$DEPLOY_INSTANCE_ID" \
                                        --query "StandardErrorContent" --output text
                                '''
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
