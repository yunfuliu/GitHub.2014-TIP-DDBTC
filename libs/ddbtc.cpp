#include "ddbtc.h"
#include <vector>

bool ddbtc::compress(cv::Mat &src,cv::Mat &dst,short BlockSize){

	if(BlockSize!=8&&BlockSize!=16){	// current version supports only 8 and 16
		return false;
	}
	if(src.type()!=CV_8U){	// should be grayscale image
		return false;
	}

	// = = = = = pre-defined data = = = = = //
	int	&imgWidth=src.cols,&imgHeight=src.rows;
	short			CM_Size=BlockSize;	// class matrix size
	const	short	DM_Size=3;			// diffused weighting size
	const	short	CM8[8][8]={	{42,	47,	46,	45,	16,	13,	11,	2},	
								{61,	57,	53,	8,	27,	22,	9,	50},	
								{63,	58,	0,	15,	26,	31,	40,	30},	
								{10,	4,	17,	21,	3,	44,	18,	6},	
								{14,	24,	25,	7,	5,	48,	52,	39},	
								{20,	28,	23,	32,	38,	51,	54,	60},
								{19,	33,	36,	37,	49,	43,	56,	55},	
								{12,	62,	29,	35,	1,	59,	41,	34}};
	const	float	DW8[3][3]={	{0.271630,	1.000000,	0.271630},	
								{1.000000,	0.000000,	1.000000},	
								{0.271630,	1.000000,	0.271630}};
	const	short	CM16[16][16]={	{6,	7	,20		,10		,53		,55		,66	,	87 ,	137,	142,	143,	144,	172,	122,	175,	164},	
								{3,	9	,23		,50		,60		,51		,65	,	74,		130,	145,	138,	148,	179,	180,	214,	221},	
								{0,	14	,24		,37		,67		,79		,96	,	116,	39,		149,	162,	198,	12,		146,	224,	1},	
								{15,26	,43		,28		,71		,54		,128,	112,	78,		159,	177,	201,	208,	223,	225,	242},	
								{22,4	,48		,32		,94		,98		,80	,	135,	157,	173,	113,	182,	222,	226,	227,	16},	
								{40,85	,72		,83		,104	,117	,163,	133,	168,	184,	200,	219,	244,	237,	183,	21},	
								{47,120	,101	,105	,123	,132	,170,	176,	190,	202,	220,	230,	245,	235,	17,	41},
								{76,73	,127	,109	,97		,134	,178,	181,	206,	196,	229,	231,	246,	19,		42,	49},	
								{103,99	,131	,147	,169	,171	,166,	203,	218,	232,	243,	248,	247,	33,		52,	68},	
								{108,107,	140	,102	,185	,167	,204,	217,	233,	106,	249,	255,	44,		45,		70,	69},	
								{110,141,	88	,75		,192	,205	,195,	234,	241,	250,	254,	38,		46,		77,		5,	100},	
								{111,158,	160	,174	,119	,215	,207,	240,	251,	252,	253,	61,		62,		93,		84,	125},	
								{151,136,	189	,199	,197	,216	,236,	239,	25,		31,		56,		82,		92,		95,		124,	114},	
								{156,188,	191	,209	,213	,228	,238,	29,		36,		59,		64,		91,		118,	139,	115,	155},	
								{187,194,	165	,212	,2		,13		,30	,	35,		58,		63,		90,		86,		152,	129,	154,	161},	
								{193,210,	211	,8		,11		,27		,34	,	57,		18,		89,		81,		121,	126,	153,	150,	186}};
	const	float	DW16[3][3]={{0.305032,	1.000000,	0.305032},	
								{1.000000,	0.000000,	1.000000},	
								{0.305032,	1.000000,	0.305032}};

	// = = = = = give cm and dw data = = = = = //
	std::vector<std::vector<short>>	CM(CM_Size,std::vector<short>(CM_Size,0));
	std::vector<std::vector<float>>	DM(DM_Size,std::vector<float>(DM_Size,0));
	if(BlockSize==8){
		for(int i=0;i<CM_Size;i++){
			for(int j=0;j<CM_Size;j++){
				CM[i][j]=CM8[i][j];
			}
		}
		for(int i=0;i<DM_Size;i++){
			for(int j=0;j<DM_Size;j++){
				DM[i][j]=DW8[i][j];
			}
		}
	}else if(BlockSize==16){
		for(int i=0;i<CM_Size;i++){
			for(int j=0;j<CM_Size;j++){
				CM[i][j]=CM16[i][j];
			}
		}
		for(int i=0;i<DM_Size;i++){
			for(int j=0;j<DM_Size;j++){
				DM[i][j]=DW16[i][j];
			}
		}
	}

	// = = = = = create Temp space = = = = = //
	std::vector<std::vector<float>>	Tempmap(imgHeight,std::vector<float>(imgWidth,0));
	for(int i=0;i<imgHeight;i++){
		for(int j=0;j<imgWidth;j++){
			Tempmap[i][j]=src.data[i*src.cols+j];
		}
	}

	// = = = = = get processing positions = = = = = //
	std::vector<std::vector<short>>	ProPo(CM_Size*CM_Size,std::vector<short>(2,0));
	for(int m=0;m<CM_Size;m++){
		for(int n=0;n<CM_Size;n++){
			ProPo[CM[m][n]][0]=m;
			ProPo[CM[m][n]][1]=n;
		}
	}

	// = = = = = get bitmap = = = = = //
	std::vector<std::vector<char>>	DoMap(imgHeight,std::vector<char>(imgWidth,false));	// it is originally the bool, not the current char, because that the process of vector<bool> is "very" slow. 



	//////////////////////////////////////////////////////////////////////////
	// initialization
	cv::Mat	tdst;	// temp dst
	tdst=src.clone();
	// = = = = = process = = = = = //
	for(int i=0;i<imgHeight;i+=CM_Size){
		for(int j=0;j<imgWidth;j+=CM_Size){

			//////////////////////////////////////////////////////////////////////////
			// calculate the local mean, maxv and minv or original image's block
			float	mean=0.;
			uchar	maxv=tdst.data[i*tdst.cols+j],
					minv=tdst.data[i*tdst.cols+j];
			short	count_mean=0;
			for(int m=0;m<CM_Size;m++){
				for(int n=0;n<CM_Size;n++){
					if(i+m>=0&&i+m<imgHeight&&j+n>=0&&j+n<imgWidth){
						count_mean++;
						mean+=(float)tdst.data[(i+m)*tdst.cols+(j+n)];
						if(tdst.data[(i+m)*tdst.cols+(j+n)]>maxv){
							maxv=tdst.data[(i+m)*tdst.cols+(j+n)];
						}
						if(tdst.data[(i+m)*tdst.cols+(j+n)]<minv){
							minv=tdst.data[(i+m)*tdst.cols+(j+n)];
						}
					}					
				}
			}
			mean/=(float)count_mean;

			//////////////////////////////////////////////////////////////////////////
			// = = = = = diffusion = = = = = //
			short memberIndex=0;
			while(memberIndex!=CM_Size*CM_Size){

				// to decide whether the prospective coordinate is out of scope or not
				int		ni=i+ProPo[memberIndex][0],nj=j+ProPo[memberIndex][1];
				if(ni>=0&&ni<imgHeight&&nj>=0&&nj<imgWidth){
					// = = = = = dot diffusion = = = = = //
					// get error	// for maintain the dot diffusion structure, here have to take DifMap into account				
					float	error;
					if(Tempmap[ni][nj]<mean){														// y = max or min (determined by the bitmap)
						tdst.data[ni*tdst.cols+nj]=minv;
					}else{
						tdst.data[ni*tdst.cols+nj]=maxv;
					}
					error=Tempmap[ni][nj]-(float)tdst.data[ni*tdst.cols+nj];						// e = v - y
					DoMap[ni][nj]=true;

					// get fm
					double	fm=0;		
					short	hDM_Size=DM_Size/2;
					for(int m=-hDM_Size;m<=hDM_Size;m++){
						for(int n=-hDM_Size;n<=hDM_Size;n++){
							if(ni+m>=0&&ni+m<imgHeight&&nj+n>=0&&nj+n<imgWidth){
								if(DoMap[ni+m][nj+n]==false){
									fm+=DM[m+hDM_Size][n+hDM_Size];
								}
							}
						}
					}

					// diffusing
					for(int m=-hDM_Size;m<=hDM_Size;m++){
						for(int n=-hDM_Size;n<=hDM_Size;n++){
							if(ni+m>=0&&ni+m<imgHeight&&nj+n>=0&&nj+n<imgWidth){
								if(DoMap[ni+m][nj+n]==false){
									Tempmap[ni+m][nj+n]+=error	*DM[m+hDM_Size][n+hDM_Size]/fm;		// v = x + e*weight
								}
							}
						}
					}
				}				
				memberIndex++;
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}