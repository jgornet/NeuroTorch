pipeline {
  agent {
    docker {
      image 'gornet/neurotorch:v3'
      args '--runtime=nvidia --shm-size 64G'
    }

  }
  stages {
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