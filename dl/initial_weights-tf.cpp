#include "FCLayer.h"
// Trained on 8k dog / no-dog binary classification for mobilenetv2 TF (bigger)

const double initial_model_bias[] = {0.00648893, -0.00568147};
const double initial_model_weights[] = {
	-0.01941384,  0.0476031 ,  0.0298625 ,  0.03822238,  0.04660718,
        0.04355364,  0.00507468, -0.04486566,  0.0505801 , -0.03314498,
       -0.03844895, -0.04785479,  0.04721685, -0.10299487,  0.04924041,
        0.03896344, -0.0155984 , -0.03552153,  0.00877467, -0.03210478,
       -0.0623063 ,  0.0294884 ,  0.05396089,  0.03480804,  0.05099688,
        0.04304802, -0.03014269,  0.06778141, -0.03994655,  0.0159669 ,
       -0.04455925,  0.03870187, -0.05213455,  0.02988108, -0.06414664,
       -0.06124357, -0.03307279, -0.02468026,  0.04661497,  0.05075194,
        0.02493061,  0.06028948,  0.01650328, -0.02036994,  0.00965887,
        0.08900326, -0.03569976, -0.01824345, -0.04529745, -0.0137379 ,
        0.03694695,  0.04320195, -0.06518926, -0.05088365, -0.05866653,
        0.0667609 , -0.04356657, -0.04780165,  0.05619225, -0.0231221 ,
       -0.00439855, -0.01333086,  0.01023564,  0.05887213, -0.04253214,
       -0.06369811, -0.03005019, -0.03800314, -0.046354  ,  0.03370296,
        0.0138441 , -0.02477389, -0.04229919, -0.04156626, -0.06731371,
        0.06353486, -0.00408653, -0.03189164, -0.03320104,  0.02296724,
       -0.02291115, -0.05700175,  0.05594226, -0.063589  ,  0.07053277,
       -0.02971158, -0.00470862, -0.04233517, -0.02264597,  0.03136633,
        0.04525413, -0.0578924 , -0.07361139,  0.01579311,  0.04284805,
        0.02539661, -0.06195647,  0.00459903, -0.01919528,  0.05297827,
        0.02514761,  0.03857039, -0.0606037 ,  0.05996364,  0.02823537,
       -0.04523895,  0.02602829,  0.04978217,  0.06208649, -0.06847608,
       -0.03323482, -0.01518696, -0.03535764,  0.09783636,  0.03491438,
        0.04188456, -0.04589304, -0.03824399,  0.07462539, -0.0430757 ,
       -0.01085083, -0.04488987,  0.05288251,  0.00600553, -0.02265222,
       -0.04063527,  0.03246104,  0.00575658,  0.00802642,  0.02158277,
       -0.0649635 , -0.02468639, -0.00086478,  0.04388597,  0.01823113,
       -0.04763087, -0.0268889 ,  0.0423763 ,  0.04825644,  0.00496022,
        0.03642153, -0.01990389,  0.00815537,  0.0360137 , -0.02768249,
        0.02103467,  0.06597339,  0.05249522, -0.02597775,  0.01311327,
        0.01212205, -0.01448329,  0.04226281,  0.04291214,  0.10635728,
       -0.10971049, -0.06333715,  0.00111766, -0.0168651 , -0.02596719,
        0.03965508, -0.01804946, -0.00949545, -0.02790044,  0.06724703,
       -0.00365801, -0.03297945,  0.01654638,  0.02617815,  0.0145558 ,
       -0.00466872, -0.04257628,  0.05322346,  0.03840927, -0.06726205,
        0.0263706 ,  0.03234496,  0.04314252,  0.05260779,  0.02840812,
        0.02837795,  0.02146392, -0.00537202, -0.03244613, -0.07008857,
       -0.006158  , -0.02471799,  0.01267   , -0.04210487,  0.0162669 ,
       -0.01089375,  0.06265704, -0.05334616, -0.06546479, -0.05248688,
        0.03599766,  0.03608617,  0.02426572,  0.01135928,  0.06895429,
       -0.05761465,  0.05129214, -0.04880933, -0.03440227,  0.02683413,
       -0.06474099,  0.05835687,  0.04515022, -0.04773273, -0.06247222,
       -0.03913182, -0.0329206 ,  0.06448265, -0.04882207,  0.02927105,
        0.00744504, -0.07698941,  0.06412448,  0.06297974,  0.01629746,
       -0.04591841, -0.00237496, -0.02462374,  0.02680013,  0.0294229 ,
        0.04169203,  0.00874824,  0.05198708, -0.03082915, -0.00899138,
        0.00282196,  0.00985911,  0.02502061, -0.01919278,  0.0606312 ,
       -0.01867905, -0.00995243, -0.0197653 , -0.0521388 , -0.00150006,
        0.06881238, -0.01797463,  0.02694333,  0.01482747, -0.01766014,
        0.03001436, -0.02584187,  0.02135394,  0.04632222,  0.05083374,
       -0.02586169,  0.0600097 ,  0.01874647,  0.02983875,  0.03639942,
       -0.00244225,  0.01079329,  0.0186576 , -0.02953556, -0.06350961,
       -0.05623279,  0.06680258, -0.00578937,  0.02977913, -0.03307135,
       -0.03312763, -0.0428742 ,  0.04460583, -0.06745898,  0.02227406,
       -0.02968422,  0.02842878, -0.03693079, -0.00141809, -0.00023577,
       -0.00193128,  0.04604951,  0.03906474,  0.01256168, -0.03239824,
        0.01652876, -0.02002045, -0.02693337,  0.05661862, -0.00557795,
        0.00040334,  0.05812417, -0.08761649,  0.06140389, -0.04117208,
       -0.03956697,  0.00841716,  0.13606738, -0.08456706, -0.07097425,
        0.04869591, -0.05290326, -0.00942631, -0.02041759,  0.03526807,
       -0.02087742,  0.03627503,  0.02579384,  0.04044347,  0.04882688,
        0.00559835, -0.03421822,  0.00114756,  0.00352337, -0.08088426,
       -0.00709341,  0.04328224, -0.04689155,  0.06268445, -0.02987786,
        0.05854912, -0.00958261, -0.05656512,  0.00120024, -0.00399539,
        0.04807   ,  0.05029665, -0.05799916,  0.06510811,  0.06714249,
        0.04446414, -0.05595196, -0.01362355, -0.04574624,  0.06801762,
        0.02201798,  0.00426285, -0.04724446, -0.00768287,  0.06265957,
       -0.02578667,  0.06838074, -0.04785394,  0.05910901, -0.03419721,
        0.04529923,  0.01093488, -0.0332409 , -0.06365469, -0.05191346,
       -0.05611806,  0.02557564,  0.00627484,  0.07466508, -0.08799859,
        0.03277774, -0.00913973,  0.02391217,  0.06064015,  0.02046012,
        0.02804657, -0.03649382,  0.02308309, -0.03875841, -0.06244161,
        0.06199941, -0.06434465,  0.01989144, -0.00243045,  0.02676685,
        0.02150549, -0.0477405 ,  0.06025407, -0.00684981, -0.00303652,
       -0.06384823,  0.08581908, -0.00293935,  0.02540188,  0.00405108,
        0.06830903,  0.06759307, -0.05017114, -0.05376486, -0.00040778,
        0.00540853,  0.01969865,  0.06680931,  0.01213865,  0.04676564,
        0.02262248,  0.05838111,  0.03798168, -0.01329417,  0.02364715,
        0.04533391,  0.0455243 , -0.00696002,  0.00460739,  0.07709394,
       -0.0678453 ,  0.06476711, -0.08614705,  0.03621051, -0.06351107,
        0.01966944,  0.03525754, -0.06702464,  0.05623586,  0.07020354,
       -0.06708279, -0.02526406, -0.03193651, -0.01867051, -0.0074732 ,
        0.04155334,  0.0314342 , -0.00192211, -0.0042497 , -0.0110746 ,
        0.04699671,  0.01070624, -0.04870502,  0.01292455,  0.04281992,
       -0.06988139, -0.00136953,  0.01033533,  0.03470539,  0.01714482,
        0.0556484 ,  0.04492196,  0.05483937,  0.06036922, -0.0522918 ,
        0.04359276,  0.04359944, -0.05913701, -0.05345074, -0.05060087,
       -0.03705129, -0.05895845,  0.06716704, -0.01908127, -0.03040168,
       -0.04060133, -0.03786191, -0.05293647,  0.02111528, -0.050136  ,
       -0.00793328, -0.00832348, -0.02924543,  0.00627239,  0.05671876,
       -0.05034775, -0.01869621, -0.06419629, -0.02112675, -0.02235554,
        0.01222532, -0.05973125, -0.02027548,  0.04771902,  0.03027288,
        0.06134284,  0.00315211, -0.01387298,  0.00457056, -0.00151871,
        0.03533636,  0.00850128, -0.00486862,  0.02156898,  0.03060846,
       -0.03531856, -0.02788565,  0.017047  ,  0.04254204, -0.02769973,
       -0.03268259,  0.05850951, -0.04213043,  0.01511348,  0.0173711 ,
        0.00446399,  0.00165296,  0.0553688 ,  0.03887649,  0.00217824,
       -0.05028196, -0.03709932, -0.01192361, -0.02451177,  0.05607729,
       -0.01276634, -0.03399861, -0.02332137,  0.01557033, -0.04782476,
        0.06868059, -0.01411789, -0.0052043 , -0.00668867,  0.03249067,
       -0.05046564,  0.010403  , -0.06567446, -0.01981836,  0.02421093,
        0.05089598,  0.03053601, -0.03325364, -0.04382952,  0.01236065,
       -0.0620137 ,  0.00083152,  0.02098082, -0.02337592,  0.07505451,
       -0.06652067, -0.14688891,  0.10401636, -0.02983734, -0.03906136,
       -0.02620117, -0.03502974,  0.05966274,  0.04877837, -0.0486411 ,
        0.0402979 , -0.02853127,  0.00906489, -0.04832944,  0.03677183,
        0.03394945, -0.00194811,  0.02734275,  0.00885359, -0.05755546,
        0.02877956, -0.05365046,  0.0629429 , -0.03072947, -0.06590137,
       -0.02280973,  0.05485445, -0.00507681, -0.0435188 ,  0.06333317,
       -0.02987315,  0.01073673,  0.00969682, -0.00111188, -0.03623847,
       -0.02294149, -0.01552683, -0.01592159, -0.0659657 ,  0.0320628 ,
        0.00689639,  0.0145568 ,  0.00790545, -0.01711504, -0.05847983,
        0.03495919,  0.01673048,  0.02835164, -0.00244498,  0.06349193,
       -0.09738807, -0.00123312,  0.02049048,  0.00925333,  0.04903376,
       -0.01383453, -0.0346088 , -0.02729982,  0.04712782,  0.01186089,
       -0.08731174,  0.00518238, -0.05749321, -0.03750413,  0.02727388,
       -0.05618819,  0.03290688, -0.01751119,  0.03716859, -0.07376887,
        0.02930806,  0.03709141,  0.04952715,  0.04364425, -0.03370488,
        0.06285802,  0.04194481,  0.01414241, -0.00448935, -0.05454647,
       -0.01488706,  0.00950372,  0.02836654, -0.0597793 , -0.06037427,
       -0.04206377,  0.0403    ,  0.02238685,  0.03178528,  0.01992127,
       -0.02746498, -0.06172882,  0.07082575,  0.02445675,  0.05034961,
       -0.03187608, -0.03508488,  0.02492855, -0.01624819,  0.04197565,
       -0.05210487,  0.01508585, -0.0482714 , -0.06558595, -0.0375355 ,
       -0.04360026,  0.0164436 ,  0.07532851,  0.02153997,  0.01048517,
        0.01645101,  0.06024867,  0.01784074,  0.01413689, -0.00999213,
       -0.03687781,  0.03811228, -0.05906983,  0.01997925,  0.02051492,
       -0.00795908,  0.0460727 ,  0.03371346,  0.03208011, -0.02919457,
       -0.03161083,  0.0305746 ,  0.03625949,  0.05385059,  0.01846606,
       -0.00615913, -0.01321805, -0.02640109,  0.02297552,  0.01330668,
        0.0389728 , -0.01461602, -0.02445359, -0.02427669, -0.06239524,
        0.0027527 , -0.01095968, -0.03689357, -0.04000016, -0.06436425,
       -0.06614634,  0.04423966, -0.00522298, -0.0528272 , -0.00738183,
       -0.02411138, -0.03621422,  0.06310067, -0.0529613 , -0.02072072,
        0.03444595,  0.05994859, -0.05944688, -0.06583381,  0.04576071,
        0.05318722,  0.01397539, -0.06686559,  0.07852612, -0.1168083 ,
        0.05445972,  0.04781254, -0.01131208,  0.04212692,  0.04342089,
       -0.07531524,  0.04986954,  0.01967918, -0.02183267, -0.00594547,
       -0.01985594, -0.04083811, -0.04448171, -0.01298633,  0.06811661,
        0.00602243, -0.05726354, -0.01095997, -0.03823443,  0.02705491,
        0.02576202, -0.06325532,  0.00216913, -0.01863122,  0.02822733,
       -0.09067449,  0.03643541,  0.04081218, -0.06479231,  0.07557567,
        0.00437216,  0.00914436, -0.00004319,  0.05781807, -0.00073087,
        0.03261652,  0.00852614, -0.00277265,  0.03107454,  0.04548024,
       -0.00569998, -0.013739  , -0.04834165, -0.05857789,  0.0005748 ,
       -0.04972547, -0.0054825 ,  0.01391421,  0.05398742, -0.00647681,
        0.01427791,  0.02633826, -0.06296331, -0.06474841, -0.06391088,
       -0.05723842,  0.05288076,  0.02104766,  0.06993482,  0.01260803,
        0.04591008,  0.05344547,  0.05508522, -0.01757111, -0.05060376,
       -0.01029634, -0.06906761,  0.06568859, -0.02470003, -0.03310102,
        0.03260288, -0.06428549,  0.03422065, -0.05944029, -0.0155997 ,
        0.01363085, -0.06744471, -0.06097921,  0.12332154, -0.06591416,
        0.06100615,  0.02751329,  0.02665718, -0.03446124, -0.00931797,
       -0.02896199,  0.06057632,  0.04311397,  0.06042503, -0.03878371,
        0.04258091,  0.03886941,  0.04232416, -0.04002864,  0.06454041,
       -0.0612364 , -0.04796249, -0.02798804, -0.04892219,  0.03422036,
       -0.01634606,  0.03967585, -0.0117987 ,  0.04655774,  0.00702386,
        0.00389548,  0.03781265,  0.03779087,  0.0044397 , -0.00093023,
        0.02727643,  0.00081754,  0.02096038,  0.02289145,  0.05111213,
        0.05207217,  0.02358377,  0.0366431 , -0.02964409, -0.03525105,
        0.04883699, -0.06281465, -0.0534437 , -0.06808174,  0.00028774,
        0.06539747,  0.01653305, -0.00236416,  0.00330826, -0.05362694,
       -0.06598569,  0.03446211, -0.02495348,  0.05784044, -0.00068267,
        0.03721822, -0.05122806,  0.02242071,  0.00356265, -0.01085394,
       -0.00972136,  0.06447805, -0.01295307,  0.01791992, -0.02574125,
        0.04132382, -0.00778249, -0.023912  ,  0.06672364,  0.05593127,
       -0.06235393,  0.03914005,  0.03234974, -0.05754293,  0.00621852,
        0.01886808, -0.05845624, -0.03203446,  0.06217138,  0.04020861,
       -0.00312823, -0.04720695, -0.01944814, -0.03177226,  0.05662381,
        0.01547651, -0.04792424,  0.05450705, -0.06627328,  0.03093878,
       -0.19627866,  0.15431845, -0.02014265, -0.05749841, -0.07278218,
        0.05068826, -0.02373857,  0.00356465, -0.03191742,  0.0491977 ,
       -0.01891215,  0.02992712,  0.01491344,  0.01289831, -0.02343306,
        0.03743605,  0.06165548, -0.03727448, -0.03284324,  0.01235098,
        0.06407179,  0.00412874, -0.03025929, -0.01214586, -0.01129147,
        0.01812196, -0.01925188, -0.04105419,  0.03361705, -0.02781062,
       -0.0501468 , -0.02379607, -0.03468509,  0.04880198, -0.06579376,
        0.06412765, -0.04157974,  0.04398203,  0.06352506, -0.0366201 ,
       -0.05583612, -0.06210455,  0.02478166,  0.06103971, -0.04601789,
        0.03653393,  0.02618134,  0.02678289, -0.06243186,  0.01388037,
       -0.05409492, -0.01599093,  0.02498803,  0.02186118, -0.04884877,
       -0.04616123,  0.00347751,  0.06170193,  0.04594371,  0.00682709,
       -0.00023087, -0.07100466, -0.02733351, -0.03733154, -0.05549316,
        0.05936876, -0.0099268 ,  0.03175348,  0.06790185,  0.06119166,
       -0.00885411,  0.05528786, -0.00072466, -0.06882036, -0.0569595 ,
        0.02197859, -0.02367777, -0.04966859,  0.02652968,  0.05813349,
       -0.01530933, -0.01735587, -0.01895148,  0.05499969, -0.05118282,
        0.01667488,  0.04509609, -0.01465186, -0.0152634 , -0.0360124 ,
       -0.01923346, -0.02472332, -0.02263193,  0.06862766,  0.06389259,
       -0.03290392, -0.06846213,  0.10070668,  0.14876759, -0.07327846,
       -0.04506866,  0.00977147,  0.02335542, -0.00136923, -0.0063618 ,
        0.04272076,  0.0138581 ,  0.01988184, -0.12771285,  0.0779747 ,
        0.05873452, -0.06408639,  0.00779633,  0.00260609,  0.04854798,
        0.03450781, -0.05544416, -0.00388085, -0.01345785, -0.07064938,
       -0.05210965, -0.01636651,  0.03870248,  0.02411825,  0.00125129,
       -0.00907881, -0.03221006,  0.03683854, -0.03321201, -0.04119922,
        0.06475337,  0.02094232, -0.06612638,  0.01202198,  0.02189862,
        0.00855309, -0.04708292, -0.04248218,  0.01420481,  0.00166761,
       -0.0172173 , -0.05267724, -0.06882934, -0.04498868,  0.03654595,
       -0.03313111,  0.0181664 ,  0.01703871,  0.00975861, -0.00005116,
        0.00993798, -0.02878315, -0.02118677,  0.05029988,  0.01173723,
       -0.04777706,  0.02933125, -0.01625746,  0.02658715, -0.03771359,
        0.03771312, -0.05368537,  0.05523114,  0.06301477, -0.05365059,
       -0.02873385, -0.02957584, -0.01830623, -0.00484419,  0.03738395,
        0.04754903,  0.06596773, -0.04127263,  0.0345943 , -0.06640635,
       -0.03484787,  0.06580271, -0.07095088, -0.0232883 ,  0.01405874,
       -0.0516165 ,  0.01820892,  0.04547253, -0.00896819,  0.01821172,
        0.04964531,  0.0428946 , -0.00508274, -0.05608513, -0.00794252,
       -0.06019582, -0.01279587, -0.0324127 , -0.02265899,  0.03253106,
       -0.01370945,  0.037301  ,  0.04669321, -0.02765907, -0.0465927 ,
        0.11377996, -0.02595795, -0.04059877,  0.01221585,  0.00433735,
       -0.0458952 , -0.03431914,  0.01465257, -0.04897807, -0.00850622,
        0.02820813,  0.06540843, -0.03379153, -0.02655874, -0.00367906,
        0.08922285, -0.05972223, -0.04660619, -0.0103836 ,  0.03686425,
        0.04887258, -0.02276429,  0.06299204,  0.05521458,  0.0071357 ,
       -0.02612385,  0.01457448,  0.00749198, -0.02019428, -0.02618674,
        0.01832691,  0.06470156, -0.02047171, -0.03939313, -0.01094613,
        0.03408745,  0.03488975, -0.04630615, -0.01180027,  0.0045795 ,
       -0.06607256,  0.03126457,  0.02275803,  0.01578734,  0.00416311,
        0.0587067 , -0.00178113,  0.00391983, -0.01253464, -0.00019781,
       -0.01488034,  0.04572421,  0.01548296, -0.05511627, -0.01282274,
       -0.04845963,  0.04559043, -0.04182143,  0.00143201,  0.06165842,
        0.01336945, -0.024469  ,  0.06195665,  0.04725662, -0.00001636,
       -0.03910448,  0.02818533,  0.03052682, -0.02527207,  0.00199999,
       -0.05884143,  0.00562906,  0.00288625,  0.01840137,  0.01364977,
       -0.06115001,  0.01910887,  0.00699897,  0.03576256, -0.04343911,
        0.04946233, -0.06699847,  0.05712191, -0.05742656,  0.01427087,
       -0.04771185, -0.0523088 ,  0.00487467,  0.01954959,  0.03879061,
       -0.00176078, -0.05777285, -0.03082349, -0.06292952, -0.06060208,
        0.03680649, -0.01064163, -0.02103343,  0.06300785,  0.05143731,
       -0.01576827,  0.03849405,  0.03058032, -0.0275193 ,  0.01233966,
        0.05420484, -0.00299051,  0.06847611,  0.09379552, -0.0143461 ,
       -0.04056479, -0.04499868, -0.02950385, -0.04915339,  0.08124604,
        0.01096343,  0.05916412,  0.06682096, -0.04707786,  0.045375  ,
       -0.00190424,  0.03396494, -0.00529282,  0.02976437, -0.05913134,
        0.03595379,  0.05290874, -0.06096133,  0.05537532, -0.02326397,
        0.00672484, -0.05162726, -0.06470368,  0.06649984, -0.00288864,
        0.0365912 ,  0.02035938, -0.0341867 , -0.01602972, -0.0280366 ,
        0.02945733,  0.06369033,  0.01558611, -0.03228229,  0.01325639,
        0.03653878,  0.02531835, -0.02320504,  0.02130306,  0.06491923,
       -0.01608957,  0.00897579,  0.02852504,  0.01315133, -0.04612217,
        0.037605  , -0.03268132,  0.03225289,  0.05424708,  0.02687643,
       -0.04428022, -0.02573719, -0.06455997,  0.00547601,  0.05273997,
        0.04632693, -0.02208341,  0.00106448,  0.01038011, -0.0733587 ,
        0.01088302,  0.05623775, -0.04525169, -0.04368372,  0.04057755,
       -0.04097413,  0.05950828, -0.07855269,  0.06894615,  0.00585576,
        0.03280834, -0.03416295, -0.01438647, -0.07710208, -0.02479848,
        0.0541686 ,  0.05104601, -0.03190261,  0.05204596, -0.04709838,
        0.00526944,  0.03841765,  0.04071541, -0.00800208, -0.04913789,
        0.04322599, -0.00948494,  0.03679234,  0.06475904,  0.01268516,
        0.03543294,  0.02662362, -0.03208192,  0.01360559,  0.01347167,
        0.05083077,  0.03908963, -0.05874945,  0.00201295,  0.08736139,
       -0.04991239,  0.03463789,  0.04192743, -0.02250353,  0.04767364,
       -0.0717016 ,  0.02933512,  0.02401011,  0.00627237,  0.03651112,
        0.05731559,  0.02448419,  0.02377801, -0.02550211, -0.05464097,
        0.00865181, -0.00963231, -0.01893606, -0.02579706,  0.00270331,
       -0.04173358,  0.00820003, -0.06802454, -0.06554491,  0.03285787,
        0.01882831,  0.01418044,  0.01285878,  0.03495028, -0.03730971,
       -0.03872762,  0.06773747, -0.00918708, -0.0411633 ,  0.01982321,
        0.01595197,  0.04311102, -0.0163921 ,  0.03164456,  0.04587771,
        0.06888406, -0.03238666,  0.01639767, -0.06386072,  0.03001035,
        0.06397682, -0.03286213, -0.00478646, -0.03012095, -0.0578324 ,
        0.01570291,  0.03272966, -0.03996059, -0.01434891,  0.07722269,
       -0.12701856,  0.04513329, -0.02864057, -0.02821347,  0.0447961 ,
       -0.01837998, -0.01045857, -0.00818134,  0.06265369,  0.02966077,
       -0.04969759, -0.02922667,  0.06566869,  0.02737683, -0.0132467 ,
        0.0048615 , -0.0515371 ,  0.03660464,  0.07032139,  0.03032065,
        0.02332526, -0.00105961, -0.0995246 , -0.02593608,  0.01719245,
       -0.06382394,  0.03436301, -0.01213807, -0.0184012 ,  0.06507054,
        0.03480769,  0.02762479,  0.06621945,  0.0231453 , -0.00403796,
        0.05134154, -0.06262354, -0.04329694,  0.00519952,  0.0062305 ,
        0.00847126,  0.02239026,  0.05133812, -0.06553169,  0.021739  ,
        0.04608981, -0.0714247 , -0.00137609, -0.06617973,  0.04924308,
       -0.00430829, -0.03826566, -0.00959688, -0.05500911, -0.06268901,
        0.02626687,  0.01920093, -0.05995956, -0.00830557, -0.0647236 ,
        0.02880535, -0.03912794, -0.00326427, -0.03400759,  0.017843  ,
        0.01900542, -0.00763494,  0.00401761,  0.02682927,  0.05838299,
       -0.05914605, -0.05502928,  0.06166794,  0.04422006,  0.01394268,
        0.00518539, -0.02950353,  0.04027512, -0.06177088, -0.01166242,
       -0.05007046, -0.0004316 , -0.0590394 ,  0.01053009, -0.02388377,
       -0.00332742, -0.00935765,  0.00259803, -0.02387095, -0.03666623,
       -0.04314344, -0.05070223,  0.00653907,  0.00994287, -0.05747606,
        0.09563256, -0.07682195,  0.03573189, -0.03092782,  0.05667692,
        0.02495006, -0.05036043,  0.04538167,  0.05850068, -0.06422894,
        0.00528244,  0.01737042,  0.04035519, -0.01657359,  0.01625565,
        0.05202023,  0.00284936,  0.07409722,  0.04615503, -0.02287504,
        0.02010538, -0.00648223, -0.05431981, -0.01488682,  0.05249972,
       -0.04032542,  0.01666506,  0.05901971,  0.04635704, -0.06748419,
        0.05793591, -0.05155762,  0.03684234,  0.02961205, -0.00314389,
       -0.02571402,  0.04018439, -0.05979875, -0.01614335, -0.01949568,
        0.02026912, -0.06779066,  0.06844924, -0.00144732, -0.0085232 ,
        0.05679736, -0.03635646, -0.05512391,  0.01511327,  0.03522413,
        0.02626693, -0.05398958,  0.00489986,  0.0095015 ,  0.04354294,
       -0.03636352,  0.0361723 ,  0.04869324, -0.02509869,  0.01892416,
       -0.05502194,  0.02733233, -0.04498333,  0.00270993, -0.05735399,
        0.0652011 ,  0.03234071, -0.00126354, -0.02442075, -0.04734924,
        0.01557852, -0.03700277, -0.0311651 ,  0.05780887, -0.00311631,
       -0.05629391,  0.08221304, -0.0482527 ,  0.06203172,  0.01782097,
        0.01251227,  0.04627737, -0.00265162,  0.03787505,  0.02190615,
       -0.00453102, -0.04608792,  0.00711201,  0.01019236,  0.00432539,
        0.05181566, -0.04404956,  0.00948662, -0.02658995, -0.01344191,
       -0.00138086,  0.04663013, -0.06159146,  0.00671966, -0.05193694,
        0.03525851,  0.04194851,  0.05091126, -0.06630702,  0.04279842,
        0.02767678,  0.01218586, -0.00709813,  0.03047565, -0.05593896,
       -0.02972402, -0.00986346,  0.04223032,  0.05823544, -0.04003109,
       -0.06546483, -0.01435992, -0.01992968, -0.04904677,  0.06431048,
        0.01870062,  0.03843073,  0.01006066, -0.08199803, -0.06259397,
       -0.00026152,  0.02272802, -0.0539129 , -0.01513631, -0.00092918,
       -0.04514461, -0.04710225,  0.02419619, -0.03765239,  0.04438296,
        0.06628747, -0.03055998, -0.06513168,  0.06612539,  0.01062251,
        0.04126046, -0.01662996, -0.04150353, -0.02892033,  0.00396484,
        0.01042737,  0.07053409, -0.06046534,  0.0153932 , -0.08018183,
        0.03957307,  0.03281019,  0.02514856,  0.02972541,  0.05825002,
       -0.01738671,  0.01924407, -0.05390451,  0.04273545,  0.01472646,
        0.04122614, -0.04875172,  0.01761321, -0.0437477 ,  0.07156035,
        0.00774825,  0.02254007, -0.06608334, -0.0059843 , -0.06682243,
       -0.01814181,  0.01884326, -0.04657614, -0.00047934,  0.03730489,
       -0.01098294,  0.06033954,  0.0201113 , -0.02653508,  0.01642169,
       -0.03788282, -0.04728909,  0.03386112, -0.02848209,  0.06778788,
        0.04771558,  0.00332653,  0.04350983,  0.02163663, -0.02688846,
       -0.00375876, -0.06141228, -0.03427226, -0.03691399,  0.02956435,
       -0.01029424, -0.03459615, -0.0119798 , -0.03743292,  0.00803685,
       -0.02421444,  0.0110672 , -0.04607172,  0.00840098,  0.00725434,
        0.00644393,  0.03008981,  0.02348483, -0.01670847,  0.00029199,
        0.05310573, -0.04156129, -0.00194259,  0.03031809,  0.0406413 ,
       -0.04701898, -0.00213524,  0.00504766, -0.06255586,  0.00340288,
       -0.01324999, -0.04775864,  0.0353759 ,  0.007632  ,  0.03459849,
       -0.02055291, -0.05867467, -0.01622813, -0.00880391,  0.04774972,
        0.04978495, -0.04225152, -0.06529452,  0.01134055,  0.06185089,
       -0.05468683, -0.02999579,  0.03199399, -0.03047427, -0.02301367,
        0.03101248, -0.06666724,  0.06168355, -0.0238787 , -0.03784952,
        0.01745966, -0.04246125,  0.03709342,  0.00149348,  0.02857071,
        0.06329176,  0.05028782,  0.03482096,  0.01094167, -0.00579656,
        0.03726537, -0.03909887,  0.00434928,  0.04497571,  0.01905817,
        0.02797348, -0.04117005, -0.00945113, -0.00558189, -0.04058948,
        0.00830164, -0.03989986,  0.04856103, -0.01479468,  0.04217277,
        0.0107957 ,  0.03272314,  0.06957599,  0.00762267,  0.02312312,
       -0.05743381, -0.01018962,  0.04751018,  0.0562039 , -0.0165612 ,
       -0.00097917, -0.05222182,  0.04282621, -0.03763011,  0.00604591,
       -0.04543622,  0.02426337, -0.03159677, -0.02368416,  0.03848948,
       -0.10103186,  0.00450099,  0.07230315, -0.03876424, -0.0620911 ,
       -0.02335787,  0.06329001,  0.01353174, -0.05164851,  0.01804591,
       -0.04384248,  0.00351542,  0.05588349,  0.00205899,  0.03627877,
        0.00511532,  0.03993589, -0.04149386, -0.03727395,  0.06920044,
        0.04745755,  0.06464496,  0.00801315, -0.06074411,  0.04253433,
        0.02890599,  0.06228751,  0.03041296, -0.00109225,  0.00328433,
       -0.00580313,  0.01033498,  0.03631278, -0.05130141,  0.07192075,
       -0.04604426,  0.0182986 , -0.0562093 ,  0.0347136 , -0.02367371,
        0.01275957, -0.03333154, -0.05587558,  0.013206  ,  0.00972108,
       -0.05549625,  0.05268919, -0.03868099, -0.04939188, -0.0053107 ,
       -0.00677502, -0.06651254,  0.01722073, -0.03544858,  0.00387506,
       -0.01064671, -0.04682684, -0.0266066 , -0.11038934,  0.02399261,
        0.04386417,  0.01186044,  0.05132531, -0.10070708,  0.04909981,
        0.03940227,  0.0593195 ,  0.0330184 ,  0.00084838, -0.05178726,
       -0.05333521,  0.00896929,  0.00930044,  0.04133002, -0.04286458,
       -0.05273309, -0.01952375, -0.02660382,  0.05255044,  0.03539275,
        0.00230494, -0.0083438 , -0.01582603, -0.00694426,  0.01440206,
       -0.06308226, -0.01794695,  0.02724805, -0.03279144, -0.06025077,
       -0.01855956,  0.02674849,  0.04399894,  0.04898839, -0.03206191,
       -0.03723046, -0.05103385,  0.02461224,  0.02345178,  0.01180003,
        0.01255293,  0.00665463,  0.02167822,  0.00886971, -0.0412473 ,
       -0.01428443,  0.01118387, -0.06487269,  0.07681141, -0.02810742,
       -0.00418881, -0.01096261,  0.00971874,  0.01725155, -0.01181748,
        0.01287771, -0.04160441, -0.06136854, -0.08792829,  0.02888683,
        0.0644709 , -0.03344246,  0.00095249,  0.04798255,  0.05151098,
       -0.04851098, -0.02681894,  0.0192916 , -0.06790174, -0.05488878,
       -0.03317264,  0.05012712, -0.06767493,  0.0506533 ,  0.03062677,
       -0.00224754,  0.06133849, -0.00698336, -0.03333997, -0.08133452,
        0.04755632,  0.06773264,  0.04330069,  0.04337592, -0.05208118,
       -0.00774577, -0.06755836, -0.02192963,  0.04097759, -0.04813553,
       -0.00523526,  0.01957539, -0.0431776 , -0.00590471, -0.00273453,
       -0.01259038, -0.06265265, -0.02843539,  0.06646322, -0.02709941,
        0.04795906, -0.02407552, -0.04426851,  0.0372403 , -0.02654618,
       -0.00054532,  0.05690612, -0.0189786 ,  0.03228619,  0.03684258,
       -0.00711272,  0.04531145, -0.00507537,  0.03082581, -0.05858188,
        0.01463076, -0.09742197,  0.12056359, -0.03997636, -0.03371284,
        0.03696728,  0.04639718,  0.01641134, -0.04568882,  0.08127493,
       -0.01639556,  0.05919011, -0.02698306,  0.0575822 , -0.06146983,
        0.05209994, -0.00684202,  0.05307084,  0.06514233,  0.01165043,
       -0.08123226, -0.04752878,  0.04085689, -0.00733054, -0.05058082,
       -0.01582982,  0.02999375,  0.06034243,  0.05290065, -0.02150147,
       -0.00516271,  0.0006367 ,  0.06340954,  0.02101723,  0.03608696,
        0.12716854, -0.06954274,  0.00498957, -0.06043737, -0.02991045,
        0.02545174,  0.06783404, -0.03292327,  0.06369501,  0.0542772 ,
        0.02808634,  0.01210699,  0.05043261, -0.0218094 , -0.02822572,
        0.07315832, -0.05234839, -0.05191224,  0.04758714, -0.05407082,
        0.06624044, -0.00297444,  0.02982051,  0.00759531,  0.06983011,
       -0.00568381, -0.04121386, -0.05103224, -0.06216747,  0.01436137,
        0.00988469,  0.01476839,  0.06856439, -0.09115243, -0.00181973,
        0.03899467, -0.06594252,  0.00987191, -0.03888999, -0.00244   ,
       -0.02145542, -0.03405963,  0.05374214, -0.06661541, -0.03484134,
        0.04984928,  0.02500274, -0.03852868, -0.04284246,  0.03152711,
        0.01478404, -0.0004329 ,  0.04313485, -0.02099737,  0.02379597,
        0.06065112, -0.01876701, -0.01779253,  0.00174095, -0.0030277 ,
       -0.04955943,  0.06736512,  0.03280149, -0.02222412, -0.03768713,
       -0.01032907,  0.03146275, -0.00368851,  0.05019698, -0.04279177,
        0.04339004,  0.01248923, -0.05491551,  0.01266336,  0.04337666,
        0.01169537, -0.0170271 , -0.03732824, -0.00867971,  0.03609132,
        0.02044067,  0.04202714,  0.05794531,  0.04548636,  0.00505066,
       -0.0643286 , -0.032212  ,  0.06966048,  0.056775  ,  0.02609671,
        0.03102969, -0.00017559,  0.04976559, -0.00423859, -0.05618106,
       -0.00979015, -0.00220914,  0.0800769 ,  0.04110235,  0.05206806,
        0.00164099,  0.06731907,  0.03586988, -0.03157615, -0.02437182,
       -0.00757286, -0.01312124, -0.00410253,  0.02606189,  0.01375332,
        0.06883099,  0.02667042, -0.05664515, -0.03407398,  0.00262081,
        0.06453852, -0.06175946, -0.06660526, -0.02498931,  0.01722614,
        0.0495645 , -0.05507576,  0.02591731, -0.029467  , -0.00497097,
       -0.02138856, -0.01276219,  0.00338891, -0.05820258,  0.04123626,
        0.03010833,  0.01015845, -0.00037384,  0.05324573,  0.02291329,
       -0.06406698,  0.01451232, -0.01340622, -0.01943724, -0.02127831,
       -0.05972895,  0.05127369, -0.04280222,  0.05109314, -0.00146739,
        0.00485588, -0.05323514,  0.03778055, -0.04110082, -0.01828395,
        0.05675427, -0.0093194 , -0.04479589,  0.00662033,  0.05213344,
       -0.00431312,  0.05932042,  0.01709439,  0.01254485,  0.027965  ,
        0.0083271 , -0.02921547, -0.00458937,  0.00290771, -0.01145209,
       -0.00682539, -0.06073031,  0.03186868, -0.03601205,  0.0006907 ,
       -0.00027151,  0.06768876, -0.03651558,  0.06691493,  0.06167243,
       -0.06740176,  0.03964156, -0.02765979,  0.06388654, -0.03257084,
        0.01951571,  0.00444024, -0.03190737, -0.00365245, -0.00148176,
        0.00717832,  0.05492849, -0.05327775, -0.00051149,  0.01321719,
        0.01802751,  0.01599371, -0.04511372,  0.06484114,  0.06222736,
       -0.04081795, -0.01319137, -0.03582049, -0.03349021,  0.05513786,
        0.02511349,  0.01843738,  0.051626  , -0.00408929, -0.05472676,
        0.06526166, -0.0274758 , -0.00299344,  0.03498773,  0.03712191,
       -0.02889561,  0.05981804, -0.0139464 , -0.0079202 , -0.06386793,
        0.01485208, -0.05567631, -0.0453067 ,  0.06562882,  0.06705812,
       -0.04498954, -0.03886271,  0.08954875,  0.00412196,  0.10991924,
       -0.11526551,  0.03544639,  0.01451105, -0.01572122, -0.03950836,
       -0.01596118, -0.00834609, -0.06283663,  0.01239747,  0.04298965,
       -0.05045271, -0.0561244 ,  0.03466311, -0.08695589,  0.03723248,
        0.05999022, -0.06585242,  0.05577141, -0.0323137 , -0.00761056,
        0.00562361,  0.02781991, -0.0523842 ,  0.02984317,  0.01281742,
        0.01799915, -0.00998253,  0.00056352,  0.06525707,  0.06532745,
       -0.03797824,  0.06972547, -0.04711807, -0.00485837, -0.00738195,
        0.00712859, -0.01154063, -0.02071922, -0.00941824, -0.03235311,
       -0.05471677,  0.03523043,  0.05411023, -0.05800799, -0.07667347,
       -0.0444342 ,  0.02244449, -0.06070982,  0.00783317, -0.0036308 ,
        0.00220663, -0.02205217, -0.04611495, -0.02865718, -0.01371568,
        0.0179687 , -0.04209574, -0.02846952,  0.00644025,  0.06661946,
        0.00248796, -0.03116801,  0.00277999, -0.05376041, -0.06126303,
       -0.03497152,  0.0050343 ,  0.03804522,  0.0155976 , -0.00221785,
       -0.01277613,  0.05872145, -0.013387  , -0.01360778,  0.02797137,
       -0.01855378, -0.04085705,  0.03874022, -0.06265974, -0.03605475,
        0.02733957,  0.03098619, -0.03453399,  0.02827025,  0.0633958 ,
       -0.046775  ,  0.03616063,  0.02849925,  0.06730273, -0.02106797,
        0.02380784,  0.02432388, -0.03530243, -0.06915204,  0.06500959,
       -0.06230417, -0.05680612,  0.02036783,  0.02490705,  0.01052401,
        0.04334228,  0.04129041,  0.03670533,  0.0389379 , -0.00234836,
       -0.04046987, -0.0497517 , -0.0278605 ,  0.019656  , -0.03519723,
       -0.04268003, -0.04258221,  0.03129869,  0.06318725,  0.03779962,
        0.01875339,  0.03261272,  0.01721383, -0.01065431,  0.04219908,
        0.03843583,  0.02622177,  0.0239823 , -0.05225901,  0.0655719 ,
       -0.01482632,  0.04160038,  0.06635834,  0.05108767,  0.01268454,
       -0.01020189,  0.05993786, -0.00081114,  0.01296515, -0.03607095,
       -0.04870106,  0.05366134,  0.02318997,  0.02431056,  0.05882699,
        0.06246834,  0.05160049, -0.02588562, -0.04122961, -0.03972719,
       -0.00769375, -0.02739887,  0.02058423,  0.0233749 ,  0.0506798 ,
        0.04535188, -0.00911613, -0.03453955, -0.04163676, -0.03832734,
        0.03448927,  0.00004573,  0.0020686 ,  0.05511083, -0.00803703,
       -0.00745852,  0.06604587, -0.0302797 ,  0.03308413,  0.0053545 ,
        0.04174506, -0.06271707, -0.04995401,  0.00999201, -0.00002874,
       -0.0164462 , -0.03763936, -0.00446663,  0.01573764, -0.00962535,
        0.0499926 ,  0.02583632,  0.05638292,  0.04171367,  0.04461965,
       -0.01272923, -0.04236249,  0.03781992,  0.05033194,  0.05665652,
       -0.06496809,  0.04103706, -0.00590903, -0.02200676,  0.00793346,
       -0.00989033, -0.03701229, -0.03686507,  0.00379648, -0.05955226,
        0.10371853, -0.11635003, -0.0365876 ,  0.00124848, -0.04606334,
        0.0569279 ,  0.02000001, -0.01124378,  0.01959803,  0.017945  ,
        0.00600926,  0.03354189, -0.03743042, -0.01246876, -0.00838091,
       -0.01418813, -0.01515555,  0.05376597, -0.05599217,  0.03328748,
       -0.03400457,  0.04951806,  0.0539257 , -0.03668801, -0.02119984,
       -0.03167516,  0.02617301,  0.03445424,  0.03365356, -0.04291848,
       -0.04919071, -0.02036431, -0.0194801 ,  0.02315729, -0.06588472,
       -0.05562704, -0.06252351, -0.04809695,  0.04355451, -0.05105691,
       -0.00297963,  0.05675597,  0.04723665,  0.02574135, -0.03706079,
       -0.00487883, -0.01758861,  0.02951699, -0.04688271,  0.04552395,
       -0.02184469,  0.04882554, -0.06376846, -0.00295262,  0.02249434,
        0.06807406, -0.05287405, -0.01826348,  0.01680932,  0.02738042,
        0.02377375, -0.10790387,  0.03962298,  0.03213146,  0.01359431,
       -0.05147845, -0.05489923,  0.04855705,  0.01971587, -0.05776183,
       -0.02248562,  0.01269643, -0.06166355,  0.06986579,  0.00484157,
        0.02381843,  0.13598007, -0.09266935,  0.02336112,  0.0636557 ,
        0.00800769, -0.03624973, -0.0517001 , -0.05031666,  0.05177307,
       -0.00125442,  0.04307931, -0.04891125, -0.05387638,  0.05485353,
       -0.03349315, -0.05142644, -0.00252369, -0.0505454 ,  0.04258644,
       -0.06481966, -0.04136765, -0.02363673,  0.03419536,  0.04609699,
       -0.03607965, -0.06412699, -0.03324542,  0.0247523 , -0.04901236,
        0.05212911,  0.01368227,  0.01324559, -0.03956356, -0.04038132,
       -0.04979112, -0.06985275,  0.00557508, -0.02413036, -0.04048324,
        0.01166815, -0.03351574,  0.05680415, -0.02166511, -0.00765617,
       -0.07835505,  0.04975016, -0.00708968,  0.05095426, -0.04646319,
       -0.06160721,  0.01777419,  0.00548771,  0.03007667, -0.05681459,
        0.03782885,  0.01963233, -0.02976627, -0.06479575,  0.02516574,
       -0.06697289, -0.05693511,  0.00660276,  0.07389144, -0.00084789,
       -0.04621432, -0.02324738,  0.02920722,  0.02240965, -0.0145684 ,
       -0.04446549,  0.0157279 ,  0.01537352,  0.04532614,  0.02316514,
        0.01457206, -0.04179578, -0.04372599,  0.04386608, -0.00936169,
        0.04135974, -0.03353182, -0.00886408, -0.09410783,  0.02164066,
        0.0172284 ,  0.03386648, -0.06353197,  0.06529456,  0.06685153,
       -0.08091302, -0.05846215,  0.0025181 ,  0.05607835, -0.01016097,
       -0.0366791 , -0.05294175,  0.02255194,  0.06643964,  0.01826415,
        0.00533375, -0.06825135,  0.0146739 ,  0.02134066, -0.04227977,
        0.01704676, -0.02147441, -0.02021502,  0.02826082,  0.03051854,
       -0.04715114, -0.03810829,  0.06205957,  0.00159617,  0.08634974,
        0.06204603,  0.02271857, -0.06803825,  0.00107979,  0.02532646,
        0.04676452,  0.02399858,  0.05325267,  0.02189276, -0.0424851 ,
        0.00430403, -0.0105564 , -0.03738711,  0.02209406, -0.01513857,
        0.02121346,  0.04734449, -0.03766052,  0.06542675, -0.00888722,
        0.03390117, -0.01467768,  0.00129268, -0.02394437, -0.03858085,
        0.04370601,  0.04715089,  0.0391221 , -0.04576699, -0.02958987,
       -0.06430765,  0.06764092,  0.05123005, -0.05789536, -0.02154233,
       -0.06907081, -0.03449056,  0.06000991,  0.00970729, -0.03457705
}