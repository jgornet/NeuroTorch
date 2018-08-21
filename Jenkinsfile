pipeline {
    agent {
	docker {
    	    image 'gornet/neurotorch:v3'
	    args '--runtime=nvidia --shm-size 32G'
	}
    }
    stages {
        stage('Test') {
            steps {
                sh 'python setup.py test'
            }
            post {
                always {
                    junit 'test-reports/results.xml'
                }
		success {
		    archiveArtifacts 'tests/images/*tif'
		}
            }
        }
    }
}
