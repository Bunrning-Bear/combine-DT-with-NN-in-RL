1. 需要写一个循环的脚本

   - 设置独立实验次数：repeat_time 
   - 把depth和amount配置为参数
     - depth_min,depth_max: 对区间各个深度的网络进行循环遍历
     - forest_amount_min,forest_amount_max: 对区间森林的个数进行循环遍历
   - 游戏的名称也设置为参数：
   - 将结果输出

2. 需要写一个跑测试的代码：

   - 就是预测，记录reward，预测，记录reward，删掉train的过程--->我觉得写一个触发的flag就能搞定这个问题
     - 测试的周期，test_circle
     - 每次测试的time_step的个数：test_time_step

3. 需要记录的数据：

   - 每一个环境一个文件夹：游戏名称-底层网络参数-‘dt’-树的个数-树的深度 or 游戏名称-底层网络参数-baselines-树的个数-树的深度 

     - 子文件夹：用训练开始的时间戳[日期-时间]命名，用来标记某一次实验，因为我们对同一个实验环境需要训练很多次

       - 文件名：用迭代的次数命名，里面存着这一轮测试的结果：

         - episode的个数
         - 每一个episode的结果
         - 迭代的次数

         文件用pkl来存储

4. 编写读取数据，画图的脚本:

   1. 相同游戏，相同深度，不同个数的网络对比图，对比信息是average的平均值的曲线，极值的曲线[需要time_step和对应的reward]
   2. 相同游戏，不同深度，相同深度的网络对比图，对比信息是average的平均值的曲线，极值的曲线
   3. 不同游戏横向对比，尝试相同的过程
   4. 取各个游戏效果比较好的dt参数，和baselines做对比

问题分析：

1. 在训练过程累加的mean episode reward 与 打点测试得到的mean episode reward是有一定差距的，看起来差距不小。原因是前者综合了很多的次的决策网络，有一定的不稳定性。所以还是应该以后者为准。不过前者是能够看出变化趋势的
2. 即使是CartPole-v1，也存在着我们看到的波动的情况，并且目前看来波动还不小，不知道以下两者，是不是能够减小波动
   - 使用double network[已经使用]
   - 使用优先级采样队列
   - 时间队列叠加
3. 网络结构加长不一定有特别好的效果，使用了[64,8]的网络训练效果也不是特别好
4. 测试周期如果过长，打点过少，也不合适！
5. atari的游戏可以增加装饰器，之后记得加上！
   1. 有一些适用于图像中的
6. 跑完再看结果，一开始看不出什么的，不要急
7. 可以使用reshape完成数据的一维话，这样决策树的模型已经搞定了！hhh
8. 算法要初步有作用应该有300w左右的迭代，看到openai他们几乎是经过1e8这个数量级才能够趋于稳定的


## 整理一下下一步要做的事情

- 2017-7-29 完成
  1. 首先，验证当前的网络结构能够适用于很多的游戏，能够有一定的提升效果
  2. 使用300w次的迭代做测试
  3. 挑选3款在results里面做的比较好的游戏
     - BOXING
     - CrazyClimber
     - Pong
     - AirRaid


- 2017-7-30 开始，调整dt+nn的算法
  1. 首先是对ob的改造
  2. 然后分析一下dt+nn如何进行模块化，需要调整什么

- 2017-8-5，提高性能：

  按照ncalls排序

  ```
     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  24839731/32525   32.841    0.000   69.404    0.002 copy.py:137(deepcopy)
   12371584   10.437    0.000   10.437    0.000 {method '__deepcopy__' of 'numpy.generic' objects}
  95909/95905    6.795    0.000   68.335    0.001 copy.py:239(_deepcopy_dict)
   12500017    6.373    0.000    8.201    0.000 copy.py:253(_keep_alive)
   49769281    5.360    0.000    5.366    0.000 {method 'get' of 'dict' objects}
       6001    4.880    0.001    4.880    0.001 {built-in method _pywrap_tensorflow.TF_Run}
      68935    3.789    0.000    9.968    0.000 Global_Function.py:8(dic_to_list)
      70609    3.163    0.000    5.709    0.000 {built-in method builtins.sorted}
      15006    3.035    0.000    3.035    0.000 ale_python_interface.py:135(act)
   37627146    2.481    0.000    2.481    0.000 {built-in method builtins.id}
    8823680    2.470    0.000    2.470    0.000 Global_Function.py:10(<lambda>)
   22022588    1.636    0.000    1.636    0.000 {method 'append' of 'list' objects}
   12457536    1.552    0.000    1.553    0.000 {built-in method builtins.getattr}
  12401289/12401049    1.249    0.000    1.252    0.000 {built-in method builtins.issubclass}
   12339714    1.179    0.000    1.179    0.000 copy.py:192(_deepcopy_atomic)
     138942    0.966    0.000    0.966    0.000 {built-in method numpy.core.multiarray.array}
      73934    0.855    0.000    1.086    0.000 segment_tree.py:77(__setitem__)
      15004    0.802    0.000    0.805    0.000 Global_Function.py:2(list_to_dic)
  648270/97902    0.355    0.000    0.375    0.000 segment_tree.py:37(_reduce_helper)
        999    0.355    0.000    1.285    0.001 replay_buffer.py:187(update_priorities)
      31968    0.330    0.000   80.187    0.003 Agent.py:548(distribute)
       6001    0.281    0.000    7.171    0.001 session.py:892(_run)
      31968    0.255    0.000   10.460    0.000 Agent.py:745(distribute)
      31968    0.233    0.000    0.279    0.000 segment_tree.py:106(find_prefixsum_idx)
      31968    0.207    0.000   80.394    0.003 Agent.py:98(distribute)
      31968    0.185    0.000   69.280    0.002 copy.py:214(_deepcopy_list)
          1    0.168    0.168    0.168    0.168 ale_python_interface.py:132(loadROM)
      68935    0.166    0.000   10.134    0.000 Agent.py:850(filter_state)
      31968    0.148    0.000    0.901    0.000 Agent.py:866(sample_list_add_data)
          1    0.147    0.147    0.147    0.147 {built-in method _pywrap_tensorflow.TF_NewDeprecatedSession}
     602605    0.145    0.000    0.145    0.000 {built-in method builtins.min}
       5999    0.133    0.000    7.355    0.001 tf_util.py:352(__call__)
  590758/590595    0.126    0.000    0.192    0.000 {built-in method builtins.isinstance}
     178201    0.117    0.000    0.117    0.000 {method 'match' of '_sre.SRE_Pattern' objects}
       1253    0.116    0.000    0.116    0.000 {built-in method marshal.loads}
     726288    0.111    0.000    0.111    0.000 {built-in method _operator.add}
          1    0.108    0.108   97.843   97.843 Main.py:36(main)
     104147    0.105    0.000    0.105    0.000 ops.py:1276(name)
       6001    0.102    0.000    0.565    0.000 session.py:397(__init__)
        999    0.097    0.000    0.917    0.001 replay_buffer.py:128(sample)
  ```

  - 想办法删掉所有的copy
  - 尽量不要使用dic_to_list ，list_to_dic 这个方法
  - 检查哪里用了sorted这个方法

  按照cumtime排序

```
 49769281    5.360    0.000    5.366    0.000 {method 'get' of 'dict' objects}
 37627146    2.481    0.000    2.481    0.000 {built-in method builtins.id}
24839731/32525   32.841    0.000   69.404    0.002 copy.py:137(deepcopy)
 22022588    1.636    0.000    1.636    0.000 {method 'append' of 'list' objects}
 12500017    6.373    0.000    8.201    0.000 copy.py:253(_keep_alive)
 12457536    1.552    0.000    1.553    0.000 {built-in method builtins.getattr}
12401289/12401049    1.249    0.000    1.252    0.000 {built-in method builtins.issubclass}
 12371584   10.437    0.000   10.437    0.000 {method '__deepcopy__' of 'numpy.generic' objects}
 12339714    1.179    0.000    1.179    0.000 copy.py:192(_deepcopy_atomic)
  8823680    2.470    0.000    2.470    0.000 Global_Function.py:10(<lambda>)
   726288    0.111    0.000    0.111    0.000 {built-in method _operator.add}
648270/97902    0.355    0.000    0.375    0.000 segment_tree.py:37(_reduce_helper)
   602605    0.145    0.000    0.145    0.000 {built-in method builtins.min}
590758/590595    0.126    0.000    0.192    0.000 {built-in method builtins.isinstance}
572361/562501    0.061    0.000    0.068    0.000 {built-in method builtins.len}
   192836    0.026    0.000    0.026    0.000 {method 'add' of 'set' objects}
   185995    0.029    0.000    0.029    0.000 {method 'items' of 'dict' objects}
   178201    0.117    0.000    0.117    0.000 {method 'match' of '_sre.SRE_Pattern' objects}
   138942    0.966    0.000    0.966    0.000 {built-in method numpy.core.multiarray.array}
   126047    0.037    0.000    0.051    0.000 ops.py:470(__hash__)
   120435    0.027    0.000    0.199    0.000 text_format.py:1002(TryConsume)
   104147    0.105    0.000    0.105    0.000 ops.py:1276(name)
```

- 2017-8-6：
  - 八个叶子节点的时候效果很差，我要试一下算法是不是有问题，仍然用全分布来做尝试。
  - 一次分支难道只划分一个点嘛？
  - 初始化的功能要加上去，因为节点一多，我们要脱离一开始的过拟合的状态所需要的数据量就大！

- 2017-8-7：
  - 不仅仅八个节点差，现在的实验结果看来，节点的每次增加，都会导致结果变差，我要用全分布来检验我的算法的准确性：迭代次数两百万，使用4个网络节点
  - 对于算法的进一步调整（做完全分布结果，证明原本的过程是没问题之后）：
    - 一个分支可以划分多个节点，但是我们把这个放到最后再进行调优
    - 每次sample的个数保持32不变，而不是根据层数而增加。
    - 加入初始化过程。
    - 如何仍然不行：我担心这个算法效果不在于速度变快，而在于更为稳定，并且能够在后期更好的爆发:
      - 提高迭代次数进行再次测试
      - 选择换环境

- 2017-8-10:
  - 检查出问题：$s_{t}$和$s_{t+1}$ 来自不同网络节点的时候更新会不符合公式，并且增加了initial的流程
  - 目前首先进行全分布的性能测试：
    - 目前看来，8节点的性能没有原本的baseline好，即使在全分布的情况下，那么我的问题是，是不是深度越高，这个性能越差，即使是全分布？
      - 我要验证一个答案：这个性能下降的程度，是否与节点的个数有关，以此来确定下一步的改进方向！
      - 测试结果初步证实，这个差距和节点个数没有关系，不同深度的决策树表现出了差不多的训练效果，并且都比baseline略差
        - 有没有可能是初始化引起的？
        - 探索的概率中的time step应该由每一个网络节点来决定还是由整体的树来决定. 设置了全局的time_step之后，问题似乎已经得到了解决
      - baseline经过了五次实验，结果证明确实有比较好的效果，所以，不要质疑baseline的准确度


  - 接着，测试耗时，耗时测试的结果：

    ```
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     13416335 10365.093    0.001 10365.093    0.001 {built-in method _pywrap_tensorflow.TF_Run}
     24937028 5206.756    0.000 5206.756    0.000 ale_python_interface.py:135(act)
    328904512 3676.305    0.000 4567.855    0.000 segment_tree.py:77(__setitem__)
      5102611 1873.097    0.000 6458.992    0.001 replay_buffer.py:195(update_priorities)
     16634146  851.871    0.000  855.063    0.000 Global_Function.py:2(list_to_dic)
    588486269  610.153    0.000  610.153    0.000 {built-in method numpy.core.multiarray.array}
     13416335  609.037    0.000 15166.600    0.001 session.py:892(_run)
    2632836952  543.261    0.000  543.261    0.000 {built-in method builtins.min}
    163293454  440.574    0.000  701.175    0.000 Agent.py:876(sample_list_add_data)
    2712853667  359.834    0.000  359.834    0.000 {built-in method _operator.add}
    1241377598/1241377435  243.760    0.000  382.509    0.000 {built-in method builtins.isinstance}
     13416303  225.401    0.000 16044.937    0.001 tf_util.py:357(__call__)
     41613330  205.642    0.000  361.500    0.000 Agent.py:712(_get_attribute_value_for_node)
    240209471  204.967    0.000  204.967    0.000 ops.py:1276(name)
       318913  204.598    0.001 17720.492    0.056 Agent.py:243(update_to_all_model)
     13416335  203.012    0.000 1143.932    0.000 session.py:397(__init__)
      8312668  188.182    0.000 7373.202    0.001 Agent.py:107(predict)
    ```

    - 耗时操作最多的都在tensorflow那边，这里我们无力改变的。
    - update_prorities 可能能改变
    - list_to_dic和 multiarray.array 这两个函数可能能减少，想办法减少我们过程中的转化次数
    - 其他的耗时相对于整体的9小时而言，只用了10分钟不到，感觉已经没有必要调优了

- 2017-8-13

  - 目前，全分布实验已经通过，4节点的情况下已经和baseline取得了相当的效果，接下来，我们要把算法应用于真正的决策树分布中了!


### bug fix

- openai  simple.py 里面是不是有一行代码写错了？？

  ```python
                  if prioritized_replay:
                      experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(time_step))
                      (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                  else:
                      obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                      weights, batch_idxes = np.ones_like(rewards), None
                  td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards)) # 这里的ones_like(rewards) 应该是 weights
  ```

