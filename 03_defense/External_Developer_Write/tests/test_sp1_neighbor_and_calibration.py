"""Out-of-sample LOF queries must retain their actual nearest reference."""
import pytest
from External_Developer_Write.sp1_embedding_anomaly import EmbeddingAnomalyDetector


def test_unseen_query_keeps_nearest_neighbor():
    detector=EmbeddingAnomalyDetector(knn_k=1)
    assert detector._compute_lof([0.2],[[0.0],[1.0],[10.0]])==pytest.approx(0.2,abs=1e-6)


def test_reference_query_excludes_only_itself():
    detector=EmbeddingAnomalyDetector(knn_k=1)
    assert detector._compute_lof([0.0],[[0.0],[1.0],[10.0]])==pytest.approx(1.0,abs=1e-6)


def test_normal_only_calibration_retains_outlier_detection():
    from External_Developer_Write.types import DocumentEmbedding
    detector=EmbeddingAnomalyDetector(knn_k=1,threshold=0.01)
    refs=[DocumentEmbedding(str(i),[x]) for i,x in enumerate([-2.0,-1.0,1.0,2.0])]
    stats=detector.train(refs)
    validation=[DocumentEmbedding(str(i),[x]) for i,x in enumerate([-1.5,-0.5,0.5,1.5])]
    assert any(detector.detect(d.vector,stats).is_anomaly for d in validation)
    detector.calibrate(validation,stats,false_alarm_rate=0.25)
    assert sum(detector.detect(d.vector,stats).is_anomaly for d in validation)<=1
    assert detector.detect([100.0],stats).is_anomaly


def test_calibration_requires_normal_validation_data():
    detector=EmbeddingAnomalyDetector(knn_k=1)
    with pytest.raises(ValueError,match="非空"):
        detector.calibrate([],None)
