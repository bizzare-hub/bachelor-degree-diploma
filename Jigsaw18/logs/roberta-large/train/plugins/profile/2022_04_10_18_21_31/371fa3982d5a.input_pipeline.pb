$	H?B?[@?2س?:n@K?8??լ?!???4-??@	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?1?=B???1????}r??I,?,?}??r1"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails K?8??լ?1????8r?IL?'????r2"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails b??m?R??1-??;????I_{fI???r3"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)???4-??@#j??G@1E?A?e??@I,?V]?J@r4"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)?$?pt??׆?q?&T?1_%????I8fٓ?f??r5*	X9??v?@2x
AIterator::Root::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2?g?o}???!??I?grT@)?g?o}???1??I?grT@:Preprocessing2f
/Iterator::Root::Prefetch::BatchV2::Shuffle::Zip??el????!Z?e?U@)<?y?9[??14#S??M@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2q!??F
??!?AΞ?IW@)B?D???1A?܀O
@:Preprocessing2v
?Iterator::Root::Prefetch::BatchV2::Shuffle::Zip[1]::TensorSliceɭI?%r??!>?g??E@)ɭI?%r??1>?g??E@:Preprocessing2E
Iterator::Rootr?30????!l?@6?@)??3.??1??-S??@:Preprocessing2?
NIterator::Root::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSliceW|C??u??!=?? @)W|C??u??1=?? @:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::ShuffleG?ҿ$U??!?'??wV@)?WY????1???X& @:Preprocessing2O
Iterator::Root::Prefetchi??U??!5?Y????)i??U??15?Y????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?3?Ax@Qd????lX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?jB??]???z?#????!#j??G@	!       "$	?L??|yZ@?PJ!E?m@????8r?!E?A?e??@*	!       2	!       :$	????[??B??Gm	@L?'????!,?V]?J@B	!       J	!       R	!       Z	!       b	!       JGPUb q?3?Ax@yd????lX@