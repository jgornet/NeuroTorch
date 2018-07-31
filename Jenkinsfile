pipeline {
    agent {
	docker {
    	    image 'gornet/neurotorch:v1'
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
