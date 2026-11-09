# B5 원본 공간 군집 기준선

## 과정
현재 B4 최종 특징 {'extended': False, 'weighting': 'binary', 'min_df': 1, 'sublinear_tf': True, 'profile_weighting': 'positive'}를 사용합니다.
k=5·10·15·20, seed42, k-means++/n_init10/max_iter300/tol1e-4/Lloyd로 비교합니다.
같은 표본2000권(seed42)의 Euclidean silhouette와 전체 inertia를 기록합니다.
최소 군집100권인 후보 중 silhouette 최대, 동점이면 작은 k를 고릅니다.

## 결과
| k | inertia | silhouette | sample_books | min_books | max_books | unassigned_books | n_iter | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 7434.433107 | 0.051851 | 2000 | 1108 | 3924 | 0 | 17 | True |
| 10 | 6918.880016 | 0.061387 | 2000 | 387 | 1378 | 0 | 19 | True |
| 15 | 6671.180615 | 0.055768 | 2000 | 291 | 1154 | 0 | 20 | True |
| 20 | 6481.105273 | 0.052998 | 2000 | 291 | 831 | 0 | 41 | True |

선택 k=10, 원본 공간 silhouette=0.061387.
| cluster | books | top_tags | distinctive_tags |
| --- | --- | --- | --- |
| 0 | 1206 | fiction / favorites / literature / novels / classics / favourites / novel / adult | literature / classics / literary-fiction / classic / literary / general-fiction / historical-fiction / 20th-century |
| 1 | 1133 | fantasy / favorites / fiction / sci-fi-fantasy / scifi-fantasy / novels / fantasy-sci-fi / sci-fi | scifi-fantasy / sci-fi-fantasy / science-fiction-fantasy / fantasy-sci-fi / sci-fi / science-fiction / speculative-fiction / fantasy |
| 2 | 870 | romance / favorites / fiction / adult / series / contemporary / favourites / 4-stars | part-of-a-series / 4-stars / alpha-male / romance / contemporary-romance / erotica / 5-stars / adult-romance |
| 3 | 1220 | fiction / favorites / adult / adult-fiction / novels / general-fiction / book-club / contemporary | general-fiction / contemporary-fiction / bookclub / adult-fiction / book-club / literary-fiction / book-club-books / didn-t-finish |
| 4 | 938 | favorites / non-fiction / nonfiction / philosophy / reference / unfinished / abandoned / self-help | nonfiction / non-fiction / reference / philosophy / self-help / psychology / self-improvement / unfinished |
| 5 | 387 | graphic-novel / graphic-novels / favorites / fiction / comic / comics / graphic / series | graphic-novel / graphic-novels / comic / comics / graphic / comics-manga / graphic-novels-comics / comics-graphic-novels |
| 6 | 947 | childrens / children / fiction / children-s / kids / children-s-books / favorites / kids-books | kids-books / children-s-books / children-s / kids / childrens-books / childrens / children / children-s-literature |
| 7 | 1378 | fiction / mystery / favorites / mystery-thriller / thriller / suspense / crime / mystery-suspense | mystery-thriller / mystery-suspense / suspense / crime / thriller / mysteries / thrillers / mystery |
| 8 | 732 | non-fiction / nonfiction / favorites / biography / biography-memoir / memoir / adult / book-club | biography / biography-memoir / nonfiction / memoir / non-fiction / biographies / memoirs / memoir-biography |
| 9 | 1189 | ya / young-adult / favorites / fiction / teen / favourites / ya-fiction / romance | teen / ya-fiction / ya / ya-books / young-adult-fiction / young-adult / ya-fantasy / teen-fiction |

B5_assignments.csv는 전 책 배정, B5_representative_books.csv는 군집별5권입니다.
대표 책은 중심 거리순으로 저자·명시적 시리즈 중복을 제한하고, 부족할 때만 제한을 순차 완화합니다.
candidate_rank는 제약 없는 거리순 순위, selection_rule은 적용한 규칙입니다.
제목의 명시적 '#번호' 패턴만 시리즈로 처리하므로 모든 시리즈를 인식하는 것은 아닙니다.
번호는 장르 정답이 아니며, 제목·태그에 따른 해석만 가능합니다.

## 최종 군집과 구분
이 파일은 원본 공간 비교 실험 기록입니다.
실제 최종 사용 경로는 clustering.load_selected_clusters()이며 B5_improvement_selected_config.json을 읽습니다.
개선 후보가 채택되지 않으면 그 파일에도 최종 원본 설정을 명시합니다.
기본 fit_book_clusters(features,k)는 항상 원본 공간이며 최종 설정 로더를 대신하지 않습니다.
그림2개는 공용 save를 사용합니다.

수정 전 결과는 커밋 6e859f5와 그 이전 이력에 보존되어 있습니다. 현재 결과는 수정된 코드·태그 규칙으로 재실행한 값이며 이전 수치와 혼합하지 않습니다.

기존 조원 파일과 .vscode는 수정하지 않았습니다. 노트북은 실제 데이터로 다시 실행했고, 공용 코드 테스트는 작은 인공 데이터로 검사합니다.
