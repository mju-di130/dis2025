from sdv.datasets.demo import download_demo
from sdv.lite import SingleTablePreset

real_data, metadata = download_demo(
    modality='single_table', # 단일 테이블
    dataset_name='fake_hotel_guests' # 데이터명
)

synthesizer = SingleTablePreset(metadata, name='FAST_ML') # FAST_ML 프리셋 사용(성능이 최적화됨)
synthesizer.fit(data=real_data)