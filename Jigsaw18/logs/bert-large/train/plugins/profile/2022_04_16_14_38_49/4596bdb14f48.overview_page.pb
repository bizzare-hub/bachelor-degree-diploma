?$	m??˾Z@?(???m@??6?4D??!??ͮ?@	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails m?M??1?^Pj??I?q?
??r1"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails /?????1?h8en???I?<,Ԛ???r2"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)??ͮ?@ط????@1τ&?e]?@I??3?l@r3"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)???bc^??)???^R?1Mjh???I?.?.??r4"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??6?4D???R\U?]??1T?qs*??r5*	
ףp=??@2x
AIterator::Root::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2????????!??l%S@)????????1??l%S@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2???"1???!I??k?V@)^f?(???1Bd?T??@:Preprocessing2f
/Iterator::Root::Prefetch::BatchV2::Shuffle::Zip??.????!Z???T@)???[??1yڝ?/?@:Preprocessing2?
NIterator::Root::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice?4f??!X???Y@)?4f??1X???Y@:Preprocessing2v
?Iterator::Root::Prefetch::BatchV2::Shuffle::Zip[1]::TensorSlice??/??!?]ʀ@)??/??1?]ʀ@:Preprocessing2E
Iterator::Root'/2?F??!I??_j?@)JC?B?Y??1S܍?#@:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::ShuffleU1?~???!CQ??nU@)/PR`L??1@???7@:Preprocessing2O
Iterator::Root::Prefetch׿?3??!??sĵ @)׿?3??1??sĵ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?nɈ5? @Q???S?zX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2(."?????vJ?Q??!ط????@	!       "$	{O@n0Z@
PA??Em@T?qs*??!τ&?e]?@*	!       2	!       :	???#?j?????!L?@!??3?l@B	!       J	!       R	!       Z	!       b	!       JGPUb q?nɈ5? @y???S?zX@?"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMul$?Іk?!$?Іk?0"Y
;model/bert/encoder/layer_._18/output/dense/Tensordot/MatMulMatMul H.?k?!"|?Cy{?0"Y
;model/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMul?7la6k?!?Z:
P??0"Y
;model/bert/encoder/layer_._17/output/dense/Tensordot/MatMulMatMulP	1?-k?!`?????0"Y
;model/bert/encoder/layer_._12/output/dense/Tensordot/MatMulMatMull?,??k?!??Dj???0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMul??Pêk?!?0C?N??0"Y
;model/bert/encoder/layer_._20/output/dense/Tensordot/MatMulMatMulR?Pk?!s-?i???0"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMul?8 ?[k?!+zQ#5??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul?\%
k?!+?g?s??0"Y
;model/bert/encoder/layer_._22/output/dense/Tensordot/MatMulMatMul?>?(_k?!???&????0Q      Y@Y&?????a?<????X@qP?d??-@y=	`畑??"?

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
Refer to the TF2 Profiler FAQb?14.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 