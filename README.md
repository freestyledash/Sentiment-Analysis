# 细粒度用户评论情感分析
在线评论的细粒度情感分析对于深刻理解商家和用户、挖掘用户情感等方面有至关重要的价值，并且在互联网行业有极其广泛的应用，主要用于个性化推荐、智能搜索、产品反馈、业务安全等。

## 依赖
- Python 3.5
- PyTorch 0.4

## 数据集
使用 AI Challenger 2018 的细粒度用户评论情感分析数据集，共包含6大类20个细粒度要素的情感倾向。

### 数据说明
数据集分为训练、验证、测试A与测试B四部分。数据集中的评价对象按照粒度不同划分为两个层次，层次一为粗粒度的评价对象，例如评论文本中涉及的服务、位置等要素；层次二为细粒度的情感对象，例如“服务”属性中的“服务人员态度”、“排队等候时间”等细粒度要素。评价对象的具体划分如下表所示。

![image](https://github.com/foamliu/Sentiment-Analysis/raw/master/images/PingJiaDuiXiang.JPG)

每个细粒度要素的情感倾向有四种状态：正向、中性、负向、未提及。使用[1,0,-1,-2]四个值对情感倾向进行描述，情感倾向值及其含义对照表如下所示：

![image](https://github.com/foamliu/Sentiment-Analysis/raw/master/images/QingGanQingXiang.JPG)

数据标注示例如下：

![image](https://github.com/foamliu/Sentiment-Analysis/raw/master/images/ShuJuShiLi.JPG)

请到[官网链接](https://challenger.ai/dataset/fsaouord2018)下载数据集。



## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

### 使用训练模型就行预测
下载 [预训练模型](https://github.com/foamliu/Sentiment-Analysis/releases/download/v1.0/BEST_checkpoint.tar) 放在 models 目录然后执行:

```bash
$ python demo.py
```

<p>
"自从成为点评VIP后，每次出去吃饭都有种使命感，要留下自己对饭店最真实的印象，供大家参考。这家西贝在大宁的大润发楼上，五楼，停车是大宁的停车场，每小时8元，可以停到地下，坐自达电梯到五楼。我们是周四晚上6点到的，没有排队。饭店提供基本款宝宝椅和宝宝餐具。我们点了西贝面筋，其实就是大拉皮；点了烧羊棒，按根数点的，肉比较嫩，可以和服务员要辣椒和孜然粉；点了莜面鱼鱼，类似面食，带汤，口味很好，宝宝爱吃；点了大盘鸡，份量一般，味道还行；还点了每次必要的牛心菜，很下饭的叶子菜。上菜速度很快，服务员也不错，基本随叫随到，最后我还特意要了份湿巾，服务员也很热情的拿过来。整体感觉不错，不过现在没有提供的瓜子了，餐前的小米粥还是有的。"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|1|-|环境|装修情况|1|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|1|-|环境|就餐空间|0|
|服务|排队等候时间|-2|-|环境|卫生情况|-2|
|服务|服务人员态度|-2|-|菜品|分量|1|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|1|
|价格|价格水平|0|-|菜品|推荐程度|1|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|1|-|其他|再次消费的意愿|1|

<p>
"停车的时候就有管理人员出来迎接指导，进店告知我们是中了霸王餐来的，服务人员很热情地给我们安排好位置。我们选了套餐一，规定是3到3.5斤的鱼，他们居然给我们选了4.2斤的，太赞了。属于东北菜系，那种大锅烧的，套餐里有6个玉米饼，工作人员建议我们做3个，剩下的做成玉米面，小朋友很爱吃，微辣，她辣到了还是说好吃。一锅的料超足，工作人员服务非常到位，经常来问咸淡如何，火候，味道，第二赞。最后说下这套餐，量很足，我们2大1小来的，吃剩了点。一条活鱼还有活虾，点了6个炖菜，在烧的过程中就闻着超香，小朋友都说好想吃了。鱼肉和虾很鲜，刚开吃觉得略咸，可能东北菜口味偏重，加了点汤底，后面吃起来就好多了。还有餐前上的一盘腌萝卜很脆很爽口。菜品很好，这是第三赞。最后再赞下和善的老板，漂亮的老板娘，还有一群服务热情又周到的工作人员。谢谢！"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|1|-|环境|装修情况|1|
|位置|距离商圈远近|1|-|环境|嘈杂情况|1|
|位置|是否容易寻找|1|-|环境|就餐空间|-1|
|服务|排队等候时间|-2|-|环境|卫生情况|1|
|服务|服务人员态度|1|-|菜品|分量|1|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|-2|
|价格|价格水平|-2|-|菜品|推荐程度|1|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|1|-|其他|再次消费的意愿|-2|


<p>
"记得上次来汇通大厦还是江边城外最红火的时候，那排队的阵势世俗罕见，今天再来发觉汇通大厦更加破旧了。。而且一楼乘电梯的时候还有一股恶臭。。先不说这些，工作日下午六点到达哥老官门口，小桌拿到16号，排队人不算少，最终等了一个半小时终于进去了。。进去关注他们公众号就可以玩一次娃娃机，抽一个素菜。。我们抽到大白菜。。外面环境一塌糊涂，店内装修倒是还不错，还算比较干净的。我们点了8只牛蛙，不太喜欢，锅底加了牛油，因为希望更辣一些。上菜速度还蛮快的，牛蛙个头很大，肉质嫩滑，超级推荐。但是总体来说辣度还不够，没有辣到爽的感觉，美中不足啊。猪脑没有了也比较遗憾，不过血和豆腐都很好吃，服务员态度也都蛮好。总的来说是一次很愉快的用餐体验哈，如果不要排队的话下次还是会再去的。"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|-2|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|-2|-|环境|卫生情况|-2|
|服务|服务人员态度|1|-|菜品|分量|1|
|服务|是否容易停车|0|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|1|
|价格|价格水平|-2|-|菜品|推荐程度|-2|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|1|-|其他|再次消费的意愿|-2|


<p>
"罗记臭豆腐。招牌上写的是久留香臭豆腐。开了很久的一家店铺。位置很好找，特步旁边的巷子。绿色的站牌。老板主要经营糖油粑粑和臭豆腐。挺干净的。
原价是五块钱一份。总共五片，老板看我有两个人。很和蔼的送了我一片。开心。很久没遇到这么豪爽的卖家了。32个赞。尤其是在团购的情况下，老板不但没有差别对待还主动送。真的感动
臭豆腐都是现炸的。旁边有四个装调料配料的碗。老板贴心询问口味，要不要辣。因为妹妹嗓子疼选的微辣
一口一片，软中带一点脆，汤汁满满的被豆腐吸在肚子里。一口咬下去，酸辣和香菜的清新刚刚好。不会觉得辣。又意犹未尽。一整片塞嘴里。享受。眯着眼特满足，还得仰着头怕汤汁从嘴巴流出来。
非常满意的一家小店。推荐。下次还来。"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|0|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|-2|-|环境|卫生情况|-1|
|服务|服务人员态度|-2|-|菜品|分量|1|
|服务|是否容易停车|-2|-|菜品|口感|0|
|服务|点菜/上菜速度|-2|-|菜品|外观|-2|
|价格|价格水平|-1|-|菜品|推荐程度|-2|
|价格|性价比|-1|-|其他|本次消费感受|-1|
|价格|折扣力度|-2|-|其他|再次消费的意愿|-1|


<p>
"山顶来过好几次了，特别喜欢山顶看风景，而太平洋咖啡据说拥有得天独厚的观景位置。这次选择的是晚上来，可是上山来还是颇费周折的，因为乘坐缆车的游客实在是太多了，最后坐巴士上来的。店里的咖啡好像还有些特色吧，选了一款薄荷摩卡还有一款橙味儿的，感觉味道还不错的，可能是因为风景给加分了吧。说道看风景，其实在屋内看白天也许还好，但晚上基本上由于玻璃反光，窗外的景色看不太清。室外的座位还好，但数量有限，而且很潮湿，那天是有雾似地。总之，所谓小憩时看看风景，还是值得的。"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|1|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|-1|-|环境|卫生情况|1|
|服务|服务人员态度|1|-|菜品|分量|1|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|1|-|菜品|外观|-2|
|价格|价格水平|-2|-|菜品|推荐程度|1|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|-2|-|其他|再次消费的意愿|1|


<p>
"下雨天，没什么吃的，跟老婆去大队长下乡改善伙食，进去服务没的说的都是吆喝着的，一人还发一个围裙，写着为人民服务，进去的装饰，打扮都是文化大革命，红卫兵的样子！点了菌菇，番茄，牛油锅，书记毛肚，乌鸡卷，猪脑，鸭血，丸子，蟹肉棒，菌菇拼盘，蔬菜拼盘，酸菜炒饭，其中香蕉飞饼让人耳目一新感觉真的不错，酸梅汁3块钱一扎很合算！味道还可以吧，老婆吃不惯辣的！我觉得还行！两个人点多了，打包了回家的！量比较多，点单要注意！"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|-2|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|1|-|环境|卫生情况|-2|
|服务|服务人员态度|1|-|菜品|分量|0|
|服务|是否容易停车|1|-|菜品|口感|1|
|服务|点菜/上菜速度|1|-|菜品|外观|-2|
|价格|价格水平|-2|-|菜品|推荐程度|-2|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|-2|-|其他|再次消费的意愿|-2|


<p>
"抽中了霸王下午茶，好开心！试吃的地方是在南海桂澜北路39度创意空间2号楼的“MUSE法式烘焙”，搭公交176在中海万锦豪园东站下车，抬头就会看见39度两个字，很显眼的，走进正门，很容易就找到了MUSE烘焙店。

店内的环境装修简约漂亮，有点小清新的气息，整个店面面积不大，就一个小厨房，一个大厅，加上一个小课室，感觉主要是私人定制蛋糕为主，而且店内就一对刚刚创业的90后小夫妻+一名蛋糕师傅，看得出夫妻俩都很用心经营~



下面逐一介绍每款蛋糕

1、Apollo热情果酸奶慕斯 1磅 188元



这款蛋糕口感比较清新，酸奶和百香果融合而成的慕斯，酸甜而不腻，加上百香果的清香，味道真的不错！最上层嘅装饰淡奶油只是薄薄的一层，不会太腻，中间嘅蓝莓夹层慕斯又带来了另一种清新的口感，味蕾实在十分享受~~个人觉得最适合怕肥又爱吃蛋糕的MM啦



2、Aphrodite 双莓慕斯 1磅 198元

 

蛋糕表面嘅那层红色的果酱是草莓+蓝莓混合而成，有点特别，酸酸甜甜的，店主介绍说是真水果做成的酱，粒粒果肉都吃的到，完全无添加，而且蛋糕上面嘅草莓好吸引啊，鲜红诱人，卖相吸引眼球之余，内涵又真系好得喔，夹层的慕斯口感软滑细腻，莓果的味道也比较浓，正~~

而且蛋糕里面嵌入了一圈草莓，真的是下足料了！！喜欢草莓的朋友不妨尝试下



3、Aglaia 西瓜草莓慕斯 2磅 288元

 

这款蛋糕比较创新，很少听说水果慕斯中用西瓜做材料的。蛋糕最上面那层是草莓酱，铺着很多西瓜粒和草莓果肉，也是下足料了！层层慕斯夹着西瓜果冻，曲奇饼底和西瓜果肉，不过吃的时候由于西瓜是一整片的，难以用叉切开来吃，只好整片西瓜放入口中，难以将西瓜和慕斯融合在一起，好像各吃各的，好像只是材料的堆砌，难以吃出融合的口感，这也是希望店家需要改进的地方。建议西瓜片弄碎，这样感觉更容易融合一些。还有就是西瓜果冻有点奇怪，西瓜味不重，还不如不放。以曲奇碎做蛋糕底是不错的尝试，不过硬硬的曲奇饼底就不要放在慕斯夹层吧，吃起来会很怪的，放在最底层即可。



4、Zeus榴莲酸奶慕斯 2磅 298元

 

蛋糕的装饰很漂亮，很喜欢上面嘅透明泡泡，甜甜的，主要是装饰蛋糕真的很好看哟，味道方面，蛋糕嘅榴莲味道比较重，喜欢吃榴莲的朋友说榴莲的味道还不够，我本身不怎么爱吃榴莲，但至少我还是能接受这款蛋糕的，榴莲果肉完全融合在慕斯里面，每口都吃的到榴莲，相信爱吃榴莲的朋友还是会比较喜欢，但建议榴莲还是要多点，点这个榴莲慕斯的朋友肯定对榴莲的分量有很大的需求。不过慕斯夹层里的那片芒果布丁有点怪，放不放都无所谓，而且芒果味不怎样浓厚。



5、Hecate 巧克力咖啡慕斯 1磅 198元

 

最后上的是这款蛋糕，好戏在后头嘛，这是老板娘重磅推荐的一款蛋糕，她自己也十分喜爱这个。首先蛋糕外形设计上采用的是棱面设计，会反光的形成反射光芒的感觉，而且这款慕斯整体上是黑色的，带给人一种高贵大气的感觉，会显得整个蛋糕很高大上的哈。而味道上真的是很棒，上层是咖啡慕斯，下层是巧克力慕斯，咖啡慕斯有点苦，但苦中又带有点甜，很过瘾~ 巧克力慕斯有点太甜了，可以适当降低甜度，和底层的曲奇饼底一起吃，风味更佳，只是曲奇容易碎，破坏口感，总体上我觉得这款是非常不错，极力推荐~~



还有这里的花茶和牛油曲奇都不错的，特别是曲奇，好香牛油味呢。

现在的话主要是私人定制蛋糕为主，禅桂可以配送，加店主微信可以在线订购，方便快捷。还有以后会陆续出新蛋糕品种，拭目以待！"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|-2|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|1|-|环境|就餐空间|-2|
|服务|排队等候时间|-2|-|环境|卫生情况|1|
|服务|服务人员态度|1|-|菜品|分量|-2|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|-2|
|价格|价格水平|0|-|菜品|推荐程度|-2|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|0|-|其他|再次消费的意愿|1|


<p>
"很早前就知道日月光那里开了家便便餐厅，就是一直没机会去，这次917点评网活动居然看到星游城也开了一家，距离近了，就过去尝尝，说实话，没有我想象中的好，就环境装修的主题特色，就是一般简餐，而且价格也不便宜，点了两份鸡翅，BBQ烤翅和辣翅味道都一般，我点了一个传统肉酱面，这端上来的盘子我真怀疑有没有洗干净，灯光照下来盘子旁边都是油渍，瞬间胃口下降，还有朋友点的咖喱料理都好稀疏，虾仁都没有外面图片上来的那么大，马桶5号就是巧克力冰激凌，味道也一般，好像马桶2-4号是黄色盘子，很大一盆～看到别人点的，还有焦糖吐司，一个字：甜
总体来说吧，主题餐厅就是装修上和菜品摆盘上多出了新意，楼下那个海盗主题餐厅也是一般～感觉不会来第二次了～还有，卫生情况值得堪忧呀～性价比不高，不做推荐"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|-2|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|-2|-|环境|卫生情况|-2|
|服务|服务人员态度|-2|-|菜品|分量|-2|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|-2|
|价格|价格水平|-2|-|菜品|推荐程度|-2|
|价格|性价比|1|-|其他|本次消费感受|1|
|价格|折扣力度|1|-|其他|再次消费的意愿|-2|

<p>
"感谢大众点评，感谢店家热心服务和丰盛招待，吃得好撑，哈哈～

【位置】他们家位于元通地铁站附近，从地铁二号线元通站出来以后从五号口出来，然后从出口方向往前走，然后上楼梯，五号口好像位于金源广场里，所以需要再出来，到江东中路元通公交车站台以后再往前走，第一个十字路口右拐，然后沿着楠溪江东路一直往前走，然后过红绿灯，再往前走，一直要走大概十分钟，然后到了那边大的商业综合体以后进去，坐电梯到负一楼，下了电梯再往右拐往前走，很快就可以看到，还是非常好找哒，奉上门面照一张，哈哈。

【口味】他们家真的口味挺不错的，是四川口味的火锅，因为是点评大V的霸王餐，所以店家招待得非常丰盛，本来还担心不够吃，后来发现这种担心纯属多余，因为一共大概有二十几种涮菜，有牛肉、鹅肠、黄喉、牛肚、百叶肚、虾滑、茼蒿、藕片、金针菇、酥肉、羊肉、山药等等，而且更赞的是每个人都配了围裙、饮料、黑糖豆花和白开水，非常贴心，食材也很新鲜，推荐黄喉和鹅肠哈。每一道涮菜上的时候服务员都会提醒大概需要涮多长时间，真的非常贴心，我们都吃得很好。

【环境】因为位于商场里面，所以整个外面环境非常好，我们有些小伙伴们到得比较早，觉得无聊还可以出去逛一逛再回来，基本上吃喝玩乐一应俱全，店里面的环境也很好，我们周五试吃的，店里面人还蛮多的，虽然人多，但不是特别吵闹，可能因为帮我们安排了有单独的半封闭式包间，所以感觉相对安静，但是唯一觉得不太满意的就是每一桌都安排的十个人，而桌子并不算特别大，个人感觉坐得还蛮挤的，有一点点影响用餐体验。

【服务】虽然抽中的霸王餐，所以店家的服务可能是不是会更殷勤一些，但是不得不客观地说他们对每一个进店的客人都很热情，都会主动打招呼，全程用餐服务员都会帮下虾滑和及时清理空盘，赞！

再次感谢点评让落落吃到这么赞的火锅，谢谢多多，谢谢带头大哥陶小丸子和兔爷，谢谢今天晚上相伴用餐的小伙伴们，再次感谢店家～"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|-2|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|-2|-|环境|卫生情况|-2|
|服务|服务人员态度|-2|-|菜品|分量|-2|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|-2|
|价格|价格水平|-2|-|菜品|推荐程度|-2|
|价格|性价比|-2|-|其他|本次消费感受|1|
|价格|折扣力度|-2|-|其他|再次消费的意愿|-2|


<p>
"一直都很喜欢吃蛋糕，在家的附近就有一家，所以经常来光顾，渐渐地就喜欢上了这家蛋糕店，我经常过来买，有时候当早饭，有时候当点心，这家店的面包跟蛋糕我都喜欢吃，刚开始是直接买，后来发现网上有优惠劵就团优惠劵，还比较便宜。前段时间大众点评推出周四半价，我觉得这是活动特别好，又实惠了顾客，又给店家招揽生意，不错，就是希望周四半价最好不要限时间或者限购，因为有时候上班时间上凑不好,这样的话下了班也能买，或者一早上班的时候买。呵呵~当然希望大众给我们多多的谋福利啦！哈哈，说回重点，面包好吃～"
</p>

|层次一|层次二|标注|-|层次一|层次二|标注|
|---|---|---|-|---|---|---|
|位置|交通是否便利|-2|-|环境|装修情况|-2|
|位置|距离商圈远近|-2|-|环境|嘈杂情况|-2|
|位置|是否容易寻找|-2|-|环境|就餐空间|-2|
|服务|排队等候时间|-2|-|环境|卫生情况|-2|
|服务|服务人员态度|1|-|菜品|分量|1|
|服务|是否容易停车|-2|-|菜品|口感|1|
|服务|点菜/上菜速度|-2|-|菜品|外观|-2|
|价格|价格水平|0|-|菜品|推荐程度|-2|
|价格|性价比|1|-|其他|本次消费感受|1|
|价格|折扣力度|-2|-|其他|再次消费的意愿|-2|