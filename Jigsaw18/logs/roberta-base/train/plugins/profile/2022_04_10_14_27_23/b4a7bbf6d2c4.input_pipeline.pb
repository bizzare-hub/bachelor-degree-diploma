$	m???1^@1Dko?p@??հߣ?!?0?1ق@	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?i4???1?7?0???I??????r1"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??հߣ?1??+ٱq?I????u???r2"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?ȭI?%??1mU?Y??IN~?N?Z??r3"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)?0?1ق@?z?<d?@1????ǝ?@I?jH?c???r4"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??p???1X?|[?Tw?I?h8en???r5*	-????,p@2f
/Iterator::Root::FiniteRepeat::Prefetch::BatchV2?Kp????!9?m?uS@)YL?Q???1(?P?M@:Preprocessing2o
8Iterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle ?Ր??ҧ?!??????1@)?Ր??ҧ?1??????1@:Preprocessing2S
Iterator::Root::FiniteRepeatL?
F%u??!?????3@)]T????1?r˲H'@:Preprocessing2]
&Iterator::Root::FiniteRepeat::Prefetch<3?p?a??!?(gA? @)<3?p?a??1?(gA? @:Preprocessing2E
Iterator::Root?_>Y1\??!'?K?(6@)X˝?`8w?1?R??I?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI E??|??Q???A?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???0P???X??1)@!?z?<d?@	!       "$	<?$?]@U????p@??+ٱq?!????ǝ?@*	!       2	!       :$	N??ĭ????l?????????u???!?jH?c???B	!       J	!       R	!       Z	!       b	!       JGPUb q E??|??y???A?X@