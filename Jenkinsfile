pipeline {
    agent {
	docker {
    	    image 'gornet/neurotorch:v1'
	}
    }
    stages {
        stage('Test') {
            steps {
                sh 'py.test --verbose --junit-xml test-reports/results.xml sources/test_calc.py'
            }
            post {
                always {
                    junit 'test-reports/results.xml'
                }
            }
        }
        stage('Deploy for production') {
	    when {
		branch 'development'
	    }
            steps {
		input message: 'Pull changes to production? (Click "Proceed to continue")'
                sh 'git pull development production'
            }
        }
    }
}
