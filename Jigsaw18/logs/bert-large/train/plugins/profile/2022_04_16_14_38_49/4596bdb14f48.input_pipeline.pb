$	m??˾Z@?(???m@??6?4D??!??ͮ?@	!       "a
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
PA??Em@T?qs*??!τ&?e]?@*	!       2	!       :	???#?j?????!L?@!??3?l@B	!       J	!       R	!       Z	!       b	!       JGPUb q?nɈ5? @y???S?zX@