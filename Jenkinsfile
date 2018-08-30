pipeline {
    agent {
	docker {
	    image 'gornet/neurotorch:v0.0.1'
	    args '--runtime=nvidia --shm-size 64G'
	}

    }
    stages {
	stage('Build') {
	    steps {
      		sh 'make -C neurotorch/reconstruction/app2/'
	    }
	}
	stage('Test') {
	    post {
		always {
		    junit 'test-reports/results.xml'

		}

		success {
		    archiveArtifacts 'tests/images/*tif'
		}

	    }
	    steps {
		sh 'python setup.py test'
	    }
	}
    }
}
