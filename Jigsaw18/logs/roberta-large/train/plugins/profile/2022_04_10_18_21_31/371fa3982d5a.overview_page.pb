?$	H?B?[@?2س?:n@K?8??լ?!???4-??@	!       "a
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
	?jB??]???z?#????!#j??G@	!       "$	?L??|yZ@?PJ!E?m@????8r?!E?A?e??@*	!       2	!       :$	????[??B??Gm	@L?'????!,?V]?J@B	!       J	!       R	!       Z	!       b	!       JGPUb q?3?Ax@yd????lX@?"[
=model/roberta/encoder/layer_._2/output/dense/Tensordot/MatMulMatMul?? a[?j?!?? a[?j?0"\
>model/roberta/encoder/layer_._17/output/dense/Tensordot/MatMulMatMul??*???j?!??.??z?0"\
>model/roberta/encoder/layer_._15/output/dense/Tensordot/MatMulMatMulu6~z?j?!?u?6???0"\
>model/roberta/encoder/layer_._22/output/dense/Tensordot/MatMulMatMul?% և?j?! ,FҊ?0"[
=model/roberta/encoder/layer_._6/output/dense/Tensordot/MatMulMatMul??C??j?!ϖ?ޠ?0"\
>model/roberta/encoder/layer_._10/output/dense/Tensordot/MatMulMatMul?!?M?j?!????
??0"w
Ygradient_tape/model/roberta/encoder/layer_._20/intermediate/dense/Tensordot/MatMul/MatMulMatMul?O?h?j?!??q?7u??0"\
>model/roberta/encoder/layer_._16/output/dense/Tensordot/MatMulMatMul???v?j?!?4$?FΚ?0"[
=model/roberta/encoder/layer_._8/output/dense/Tensordot/MatMulMatMuln????j?!?jf?9'??0"\
>model/roberta/encoder/layer_._11/output/dense/Tensordot/MatMulMatMul?X?΄?j?!??????0Q      Y@Y     ???a     ?X@q??/#6%@y?@t? ?;?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 