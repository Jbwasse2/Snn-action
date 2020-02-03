import matplotlib.pyplot as plt

#Training_LMU_spike = [0.13651375, 0.44336128, 0.6490297, 0.7212555, 0.7708182, 0.7943126, 0.8062857, 0.8147967, 0.8191053, 0.8272595, 0.82877415, 0.83468854, 0.83611214, 0.8343735, 0.83719784]
#Testing_LMU_spike =  [15.754730999469757, 28.87858748435974, 33.210280537605286, 36.651021242141724, 36.796969175338745, 38.58173191547394, 40.92929661273956, 40.06791710853577, 38.89365792274475, 40.75377881526947, 41.6542649269104, 41.72390103340149, 42.89434552192688, 40.75568616390228, 39.009082317352295, 39.009082317352295]
#
#Training_LMU = [0.8258473, 0.90305364, 0.91439277, 0.9212031, 0.92797166, 0.9311073, 0.9307884, 0.9370558, 0.94197565, 0.94339544, 0.9433347, 0.9455896, 0.9494351, 0.95129526, 0.9543208]
#Testing_LMU =  [87.66884207725525, 89.38015103340149, 89.34962749481201, 89.98779058456421, 89.0997052192688, 89.83516693115234, 88.80971670150757, 90.39892554283142, 88.80780935287476, 90.24152755737305, 90.53246974945068, 90.19383192062378, 90.61068892478943, 90.0574266910553, 90.15377163887024, 90.15377163887024]
#
#Training_SNN = [0.65374076, 0.8242567, 0.82299256, 0.82899433, 0.8737587, 0.88673013, 0.8881727, 0.87918717, 0.8643745, 0.8757896, 0.89945865, 0.9100955, 0.91205055, 0.916511, 0.9091237]
#Testing_SNN =  [80.4697036743164, 80.67097663879395, 82.14667439460754, 85.20776033401489, 84.85099673271179, 85.67517399787903, 83.79311561584473, 85.26308536529541, 84.33302640914917, 86.10443472862244, 87.00110912322998, 87.98363208770752, 87.31398582458496, 87.89396286010742, 86.40491366386414, 86.40491366386414]
#
#Training_1_deep =  [[0.563058], [0.7937697], [0.825798], [0.83527696], [0.85147136], [0.8294309], [0.8598723], [0.87473804], [0.87515944], [0.8777598], [0.8748216], [0.8715037], [0.8650313], [0.86545646], [0.8765147], [0.87861395], [0.89252687], [0.87581617], [0.8550132], [0.87026995]]
#Testing_1_deep =  [72.26133346557617, 77.00320482254028, 78.39495539665222, 80.82073926925659, 75.19650459289551, 82.16670751571655, 81.40453100204468, 81.26144409179688, 80.53361773490906, 78.84615659713745, 79.24965620040894, 79.11324501037598, 78.00480723381042, 79.81913685798645, 79.7256588935852, 79.73805665969849, 77.83024311065674, 76.26011371612549, 78.0353307723999, 79.38033938407898, 79.38033938407898]
#
#Training_2_deep = [[0.3779724], [0.7210201], [0.7886715], [0.8007205], [0.81156236], [0.81317574], [0.792293], [0.78229773], [0.7751799], [0.7490168], [0.83113533], [0.8480055], [0.85073495], [0.85549915], [0.8559888], [0.8555029], [0.85388577], [0.852819], [0.8512664], [0.8536542]]
#Testing_2_deep =  [59.82238054275513, 73.02445769309998, 75.01431107521057, 75.15453100204468, 73.9898145198822, 71.30647301673889, 72.86515831947327, 70.4708456993103, 70.26385068893433, 72.69154191017151, 72.85275459289551, 70.559561252594, 68.59356164932251, 68.39991807937622, 67.70547032356262, 67.32867956161499, 66.90705418586731, 66.1582350730896, 65.24248123168945, 64.77029919624329, 64.77029919624329]
#
#Training_3_deep =  [[0.2242962], [0.6548682], [0.74195975], [0.74666315], [0.7513514], [0.7449359], [0.73601115], [0.7335778], [0.7248087], [0.698376], [0.5618433], [0.72870356], [0.7978924], [0.800246], [0.80274385], [0.80128616], [0.8031311], [0.8018138], [0.8011343], [0.79939944]]
#Testing_3_deep = [49.151021242141724, 69.17162537574768, 70.1703667640686, 68.52392554283142, 68.68227124214172, 66.40338897705078, 67.92200803756714, 67.57478713989258, 66.71722531318665, 63.31177353858948, 32.12282657623291, 76.0111391544342, 76.58730149269104, 76.55200958251953, 76.12560987472534, 75.95199942588806, 75.47408938407898, 75.36439299583435, 74.78823065757751, 74.20921325683594, 74.20921325683594]
#
#Training_1_reconnect = [[0.085053], [0.25431624], [0.33724716], [0.39387453], [0.44292092], [0.46957755], [0.49478027], [0.5152644], [0.54209566], [0.5663417], [0.5831397], [0.6058256], [0.6346081], [0.62791926], [0.65874785]]
#Testing_1_reconnect =  [10.550213605165482, 17.126449942588806, 21.35893553495407, 25.34722089767456, 26.739928126335144, 29.62835729122162, 30.691009759902954, 33.62045884132385, 36.84466481208801, 36.90285384654999, 39.331501722335815, 42.03773736953735, 44.04761791229248, 43.12328398227692, 44.939520955085754,
#44.939520955085754]
#
#Recurrent 1 out
#Training [[0.6754711], [0.82134116], [0.847671], [0.8463564], [0.8588564], [0.8717287], [0.86774313], [0.8683739], [0.8677394], [0.8731535], [0.8794263], [0.87919456], [0.8803495], [0.87722266], [0.88474166]]
#Testing [76.48841738700867, 80.75096607208252, 80.59749007225037, 83.11969041824341, 82.45753049850464, 82.64864683151245, 82.86486268043518, 82.33687281608582, 82.44208693504333, 83.95270109176636, 82.04054236412048, 85.21139025688171, 82.97587037086487, 83.59459638595581, 84.26158428192139, 84.26158428192139]

#raining [[0.6707675], [0.8199164], [0.85408056], [0.8507637], [0.8671656], [0.8769073], [0.8687614], [0.871136], [0.86537236], [0.853769], [0.851269], [0.85207826], [0.8601178], [0.889943],
#[0.90208584], [0.90533054], [0.9108853], [0.9132637], [0.9167211], [0.9177356], [0.9192553], [0.9168731], [0.9220745], [0.9201178], [0.919829], [0.91800153], [0.91398174], [0.8919871], [0.86235183], [0.851193], [0.88155776], [0.90675914], [0.9157827], [0.9226634], [0.92524314], [0.9294035], [0.9308435], [0.9306497], [0.9331801], [0.9305471], [0.9318541], [0.9288944], [0.93173635], [0.93099165], [0.93082064], [0.9292553], [0.92956305], [0.9282713], [0.92844987], [0.92525834]]
#Testing [77.2702693939209, 81.46138787269592, 81.47007822990417, 82.70077109336853, 81.81853294372559, 81.09555840492249, 82.73455500602722, 81.04729652404785, 80.19787669181824, 80.9633195400238, 82.29247331619263, 80.85521459579468, 84.99324321746826, 86.22779846191406, 86.80984377861023, 86.8803083896637, 86.00386381149292, 85.43533086776733, 86.65733337402344, 86.3146722316742, 85.45752763748169, 85.53571701049805, 85.46911478042603, 85.72200536727905, 85.77606081962585, 84.65733528137207, 84.51351523399353, 84.56660509109497, 79.72972989082336, 82.90637135505676, 86.1515462398529, 88.07818293571472, 87.87644505500793, 87.99806833267212, 87.48552203178406, 88.2490336894989, 87.812739610672, 87.87934184074402, 87.44497895240784, 87.3552143573761, 87.59073615074158, 87.27027177810669, 87.55019307136536, 87.59845495223999, 87.72779703140259, 86.98552250862122, 86.78281903266907, 86.43629550933838, 86.8542492389679, 86.86776161193848, 86.86776161193848]
#hippo
#Training [[0.6231269], [0.8406079], [0.86329025], [0.87914515], [0.86793315], [0.85141337], [0.87581307], [0.8887158], [0.8821353], [0.8730699], [0.84697187], [0.8738336], [0.8918199], [0.90134877], [0.89964664], [0.8923252], [0.8929445], [0.8988792], [0.8988716], [0.8970783], [0.88965803], [0.890019], [0.8922758], [0.89605623], [0.90233284], [0.88099164], [0.8486018], [0.8790008], [0.89091945], [0.9056117]]
#Testing [77.22490429878235, 80.16216158866882, 80.24131059646606, 81.24324083328247, 78.02316546440125, 79.37548160552979, 80.26254773139954, 81.6969096660614, 81.66409134864807, 77.05405354499817, 78.79440188407898, 80.87741136550903, 81.24613761901855, 81.87645077705383, 81.80984258651733, 80.82721829414368, 81.82914853096008, 81.42663836479187, 81.24131560325623, 80.53764700889587, 80.44884204864502, 80.67663908004761, 79.99420762062073, 81.72876238822937, 80.57239651679993, 76.06370449066162, 76.55405402183533, 79.28860783576965, 81.75965547561646, 81.61776065826416, 81.61776065826416]
#Big recurrent 12000
#Training [[0.7248936], [0.8444491], [0.87265575], [0.8844909], [0.8459309], [0.8800114], [0.9008055], [0.9130129], [0.91732144], [0.9138906], [0.9150798], [0.91767097], [0.9116147], [0.866326], [0.8552242], [0.87283057], [0.901535], [0.9125456], [0.9251482], [0.9251596], [0.93090045], [0.9355015], [0.93891335], [0.94149697], [0.9431573], [0.9448328], [0.9451178], [0.9453192], [0.9456953], [0.94420594]]
#Testing [80.37644624710083, 80.84652423858643, 83.49517583847046, 81.81273937225342, 81.01351261138916, 83.47490429878235, 85.38127541542053, 85.423743724823, 85.43436527252197, 84.27702784538269, 84.75385904312134, 85.06563901901245, 82.94015526771545, 81.78281784057617, 75.09362697601318, 86.20752692222595, 85.72200536727905, 86.01544499397278, 86.26737594604492, 86.86293363571167, 86.46138906478882, 86.38127446174622, 86.33590936660767, 86.52509450912476, 86.25578880310059, 86.19884252548218, 86.43243312835693, 86.34845614433289, 86.70077323913574, 86.24324202537537, 86.24324202537537]
#100 size recurrent dropout
Training_drop=  [[0.13492021], [0.32011017], [0.48031154], [0.61453265], [0.61041796], [0.688484], [0.73086625], [0.7437348], [0.7545745], [0.7038222], [0.7932941], [0.7562044], [0.79518235], [0.8137576], [0.816345], [0.804073], [0.81800914], [0.82604104], [0.82786477], [0.83724546], [0.8500456], [0.82320666], [0.8501444], [0.84901214], [0.84971887], [0.84286857], [0.85830545], [0.8577964], [0.85956305], [0.85537237]]
Testing_drop= [1.0250965133309364, 17.16216206550598, 28.55212390422821, 31.589767336845398, 39.380308985710144, 41.65540635585785, 47.78185188770294, 50.59555768966675, 54.101353883743286, 56.77992105484009, 57.50482678413391, 58.49034786224365, 62.661194801330566, 61.663126945495605, 60.70077419281006, 61.87644600868225, 61.13320589065552, 62.84555792808533, 64.67278003692627, 65.1650607585907, 65.30115604400635, 65.29054045677185, 65.26158452033997, 65.92084765434265, 67.4768328666687, 66.78764224052429, 67.89575219154358, 65.79053997993469, 65.57722091674805, 67.06081032752991, 67.06081032752991]

#100 size normal
Training_no_drop= [[0.2538184], [0.78772795], [0.8601216], [0.8782257], [0.8862386], [0.891535], [0.89585865], [0.897481], [0.9006041], [0.9040046], [0.90545595], [0.9067211], [0.90859044], [0.9099468], [0.91024697], [0.91262156], [0.91476446], [0.9148632], [0.91436553], [0.91521657], [0.9169225], [0.9178913], [0.9189818], [0.91765195], [0.9190273], [0.9209043], [0.9209195], [0.9199392],[0.92100686], [0.9213754]]
Testing_no_drop= [34.416988492012024, 46.96621596813202, 50.79343914985657, 52.399611473083496, 53.66505980491638, 54.15636897087097, 54.685330390930176, 55.54150342941284, 55.19015192985535, 55.87644577026367, 56.041502952575684, 56.20849132537842, 56.062740087509155, 56.11196756362915, 56.89381957054138, 56.53861165046692, 56.995171308517456, 56.87837600708008, 56.97104334831238, 57.15057849884033, 57.37355351448059, 56.03861212730408, 56.1959445476532, 57.015442848205566, 56.72876238822937, 56.39575123786926, 57.02316761016846, 57.258689403533936, 57.270270586013794, 56.96331858634949, 56.96331858634949]



def make_plot(train_data, test_data, title):
    train_data = [x[0] for x in train_data]
    train_data = [i * 100 for i in train_data]
    plt.plot(range(1,len(train_data)+1), train_data, label="Training Accuracy")
    plt.plot(range(1,len(test_data)+1), test_data, label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("% Accuracy")
    plt.title(title)
    plt.legend(loc="upper left")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,100))
    plt.show()
make_plot(Training_no_drop, Testing_no_drop, "No Dropout 100 EnsembleArray")
make_plot(Training_drop, Testing_drop, "Dropout 100 EnsembleArray")

#make_plot(Training_LMU_spike, Testing_LMU_spike, "Spiking LMU")
#make_plot(Training_LMU, Testing_LMU, "Regular LMU")
#make_plot(Training_SNN, Testing_SNN, "EnsembleArray")
#make_plot(Training_1_deep, Testing_1_deep, '1 Deep ensembleArray')
#make_plot(Training_2_deep, Testing_2_deep, '2 Deep ensembleArray')
#make_plot(Training_3_deep, Testing_3_deep, '3 Deep ensembleArray')
#make_plot(Training_1_reconnect, Testing_1_reconnect, 'reconnect')

