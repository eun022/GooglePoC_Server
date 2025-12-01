from textwrap import dedent

EXAMPLE_BLOCK_DCIT = {
    "chart_type" : dedent("""다음 이미지는 차트입니다. 차트가 bar, line, pie, scatter, boxplot, violin, treemap, mixed 인지 구별한 후 차트 종류만 출력하세요. (mixed는 2개 이상의 차트가 결합된 형태입니다.)설명 금지. 딱 한 단어만."""),
    "convert_chart_type": dedent("""이미지에서 사용자가 변환을 원하는 타겟 차트의 종류만 정확히 추출해 출력하세요. chart는 다음 중 하나입니다: bar, line, pie, scatter, boxplot, violin, treemap, mixed. 설명 금지. 차트 이름 하나만 출력하세요."""),
    "EXAMPLE_STRUCT" : dedent("""
        ※ 공통 규칙 (중요)
        - **추출한 데이터는 일본어로 출력하라 **
        - series = 개별 엔티티, 범레에 들어갈 데이터 부분을 명시한다. 만약 차트가 단일이라면 Series에 데이터들을 넣고, categories는 ["1"]로 해둔다. 
        - categories = 공통된 기준(예: 연도, 시험번호, 항목번호)  
        - data는 series를 key로 두고 categories 순서대로 값을 배열로 넣는다.  
        - series와 categories의 역할을 절대로 뒤집지 말 것. 
        - categories가 딱히 없다면 ["1"]로 채운다. 절대 값을 비우면 안 됨. 
        - 모든 차트는 아래 공통 키를 가진다: "chart_type", "series", "categories", "data", "axes", "legend"
        - legend는 각 series 라벨을 key로 하고, 값은 내부 2×2 패턴 배열 [a,b,c,d] 이다. (바깥 테두리는 고정 1, 내부만 0/1 다양화)
        - 중요, 단일 차트(원래 범례가 없던 차트)라도 series에 항목들을 올리고, categories는 ["1"]로 둔다.
        - 패턴은 예시처럼 서로 겹치지 않도록 다르게 지정한다.
        - legend는 절대 겹치지 않는다. 우선 순위는 다음과 같다.
          [1,0,0,0,0,0] [1,0,1,0,0,0] [1,1,0,0,0,0] [1,1,0,1,0,0] [1,0,0,1,0,0] [1,1,1,0,0,0] [1,0,1,1,0,0] [1,1,1,1,0,0] [0,1,1,0,0,0] ... 
          나머지는 랜덤으로 2*3으로 변환했을때 절대 겹치지 않게 제공한다. 

        [출력 형식 예시]
        {
          "chart_type": { "type": "bar" },  #혹은 "line"
          "series": ["2008", "2012"],
          "categories": ["Gold", "Silver", "Bronze"],
          "data": {
            "2008": [7, 16, 18],
            "2012": [11, 11, 13]
          },
          "axes": { "Y": { "range": [0, 30], "interval": 5 } },
          "legend": {
            "2008": [1,0,0,0,0,0],
            "2012": [1,0,1,0,0,0] 
          }
        }

        {
          "chart_type": {
            "type": "pie"
          },
          "series": [
            {
              "name": "환경 변화 경험",
              "categories": [
                { "name": "있다", "value": 95.6 },
                { "name": "없다", "value": 4.4 }
              ]
            },
            {
              "name": "채식 실천",
              "categories": [
                { "name": "시도하지 않음", "value": 62.1 },
                { "name": "간헐적 채식", "value": 27.4 },
                { "name": "플렉시테리언", "value": 9.1 },
                { "name": "항상 채식", "value": 1.4 }
              ]
            }
          ],
          "legend": [  // key 이름이 같다면 패턴도 동일하게 하라. key 이름이 다르면 무조건 패턴도 달라야한다. 
            {
              "환경 변화 경험":  [1,0,0,0,0,0]
              "있다": [1,1,0,0,0,0] , 
              "없다":[1,1,0,1,0,0] 
              ]
            },
            {
              "채식 실천": [1,0,1,0,0,0] 
              "있다": [1,1,0,0,0,0] ,
              "없다": [1,1,0,1,0,0] ,
               "플렉시테리언": [1,1,0,1,0,0] ,
               "항상 채식": [1,1,1,0,0,0] 
              ]
            }
          ],
         
        }

        {
          "chart_type": { "type": "violin" }, #혹은 "boxplot"
          "series": ["Yes", "No"],
          "categories": ["Thur", "Fri"],
          "data": {
            "Thur": {
              "Yes": {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]},
              "No":  {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]}
            },
            "Fri": {
              "Yes": {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]},
              "No":  {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]}
            }
          },
          "axes": { "Y": { "range": [0, 30], "interval": 5 } },
          "legend": {
            "Yes":  [1,0,0,0,0,0],
            "No":  [1,0,1,0,0,0] 
          }
        }

        {
          "chart_type": {
            "type": "mixed",
            "components": [
              { "type": "bar",  "series": ["Min", "Max"], "y_axis": "left"  },
              { "type": "line", "series": ["Median"],     "y_axis": "right" }
            ]
          },
          "series": ["Min", "Max", "Median"],
          "categories": ["Product A","Product B","Product C","Product D","Product E"],
          "data": {
            "Min":    [5,10,7,12,6],
            "Max":    [95,100,90,97,93],
            "Median": [75,85,65,80,70]
          },
          "y_axes": {
            "left":  { "label": "Review Score Range", "range": [0,100], "interval": 5  },
            "right": { "label": "Median Review Score", "range": [50,100], "interval": 5  }
          },
          "legend": {
            "Min":    [1,0,0,0,0,0],
            "Max":    [1,0,1,0,0,0] ,
            "Median": [1,1,0,0,0,0] 
          }
        }

        {
          "chart_type": {
            "type": "treemap"
          },
          "series": [
            {
              "name": "아침식사",
              "categories": [        //값이 하나라면, { "name": "아침식사", "value": 18 }
                { "name": "와플", "value": 18 },
                { "name": "계란", "value": 15 },
                { "name": "팬케이크", "value": 12 }
              ]
            },
            {
              "name": "점심식사",
              "categories": [
                { "name": "샐러드", "value": 12 },
                { "name": "카스텔라", "value": 15 },
                { "name": "수프", "value": 10 },
                { "name": "파이", "value": 9 },
                { "name": "케이크", "value": 9 }
              ]
            }
          ],
          "legend": [  //"series"는 무조건 다르게
            {
              "아침식사": [1,0,0,0,0,0],
              "와플": [1,1,0,0,0,0] ,
              "계란": [1,1,0,1,0,0] ,
              "팬케이크": [1,0,0,1,0,0] 
            },
            {
              "점심식사": [1,0,1,0,0,0],
              "샐러드": [1,1,1,0,0,0] ,
              "카스텔라": [1,0,1,1,0,0] ,
              "수프": [1,1,1,1,0,0],
              "파이": [0,1,1,0,0,0],
              "케이크": [0,1,1,1,0,0]
            }
          ]
        }


        {
          "chart_type": { "type": "scatter" },
          "series": ["data1", "data2", "data3"],
          "categories": [""],
          "data": { "1": [""] },  //산점도는 데이터를 저장하지 않음
          "legend": {
            "data1":  [1,0,0,0,0,0],
            "data2": [1,0,1,0,0,0],
            "data3": [1,1,0,0,0,0] 
          }
        }
        """
    ),
    "public": dedent("""
        ※ 공통 규칙 (중요)
        - **추출한 데이터는 일본어로 출력하라 **
        - series = 개별 엔티티, 범레에 들어갈 데이터 부분을 명시한다. 만약 차트가 단일이라면 Series에 데이터들을 넣고, categories는 ["1"]로 해둔다. 
        - categories = 공통된 기준(예: 연도, 시험번호, 항목번호)  
        - data는 series를 key로 두고 categories 순서대로 값을 배열로 넣는다.  
        - series와 categories의 역할을 절대로 뒤집지 말 것. 
        - categories가 딱히 없다면 ["1"]로 채운다. 절대 값을 비우면 안 됨. 
        - 모든 차트는 아래 공통 키를 가진다: "chart_type", "series", "categories", "data", "axes", "legend"
        - legend는 각 series 라벨을 key로 하고, 값은 내부 2×2 패턴 배열 [a,b,c,d] 이다. (바깥 테두리는 고정 1, 내부만 0/1 다양화)
        - 중요, 단일 차트(원래 범례가 없던 차트)라도 series에 항목들을 올리고, categories는 ["1"]로 둔다.
        - 패턴은 예시처럼 서로 겹치지 않도록 다르게 지정한다.
        - legend는 절대 겹치지 않는다. 우선 순위는 다음과 같다.
          [1,0,0,0,0,0] [1,0,1,0,0,0] [1,1,0,0,0,0] [1,1,0,1,0,0] [1,0,0,1,0,0] [1,1,1,0,0,0] [1,0,1,1,0,0] [1,1,1,1,0,0] [0,1,1,0,0,0] ... 
          나머지는 랜덤으로 2*3으로 변환했을때 절대 겹치지 않게 제공한다. 

        [출력 형식 예시]"""),
    "bar" : dedent("""        {
          "chart_type": { "type": "bar" }, 
          "series": ["2008", "2012"],
          "categories": ["Gold", "Silver", "Bronze"],
          "data": {
            "2008": [7, 16, 18],
            "2012": [11, 11, 13]
          },
          "axes": { "Y": { "range": [0, 30], "interval": 5 } },
          "legend": {
            "2008": [1,0,0,0,0,0],
            "2012": [1,0,1,0,0,0] 
          }
        }"""),
     "line" : dedent("""        {
          "chart_type": { "type": "line" },  
          "series": ["2008", "2012"],
          "categories": ["Gold", "Silver", "Bronze"],
          "data": {
            "2008": [7, 16, 18],
            "2012": [11, 11, 13]
          },
          "axes": { "Y": { "range": [0, 30], "interval": 5 } },
          "legend": {
            "2008": [1,0,0,0,0,0],
            "2012": [1,0,1,0,0,0] 
          }
        }"""),
    "pie" : dedent(""" 
        - **추출한 데이터는 일본어로 출력하라 **
        - categories = 공통된 기준(예: 연도, 시험번호, 항목번호)  
        - series와 categories의 역할을 절대로 뒤집지 말 것. 
        - categories가 딱히 없다면 ["1"]로 채운다. 절대 값을 비우면 안 됨. 
        - 모든 차트는 아래 공통 키를 가진다: "chart_type", "series", "categories", "data", "axes", "legend"
        - legend는 각 series 라벨을 key로 하고, 값은 내부 2×2 패턴 배열 [a,b,c,d] 이다. (바깥 테두리는 고정 1, 내부만 0/1 다양화)
        - 중요, 단일 차트(원래 범례가 없던 차트)라도 series에 항목들을 올리고, categories는 ["1"]로 둔다.
        - 패턴은 예시처럼 서로 겹치지 않도록 다르게 지정한다.
        - legend는 절대 겹치지 않는다. 우선 순위는 다음과 같다.
          [1,0,0,0,0,0] [1,0,1,0,0,0] [1,1,0,0,0,0] [1,1,0,1,0,0] [1,0,0,1,0,0] [1,1,1,0,0,0] [1,0,1,1,0,0] [1,1,1,1,0,0] [0,1,1,0,0,0] ... 
          나머지는 랜덤으로 2*3으로 변환했을때 절대 겹치지 않게 제공한다.     
          pie 차트가 2개 이상 보이더라도 **절대 묶지 말고** 각각을 **하나의 series 로만 표현하라.**
          절대 리스트로 chart 객체를 2개로 만들지 말고, 항상 단 하나의 chart json 만 출력하라.
          [출력 형식 예시]   
                  {
  "chart_type": {
    "type": "pie"
  },
  "series": [
    {
      "name": "Experience of Environmental Change",
      "categories": [
        { "name": "Yes", "value": 95.6 },
        { "name": "No", "value": 4.4 }
      ]
    },
    {
      "name": "Practice of Vegetarian Diet",
      "categories": [
        { "name": "No Attempt", "value": 62.1 },
        { "name": "Occasional Vegetarian", "value": 27.4 },
        { "name": "Flexitarian", "value": 9.1 },
        { "name": "Always Vegetarian", "value": 1.4 }
      ]
    }
  ],
  "legend": [
    {
      "Experience of Environmental Change": [1, 0, 0, 0, 0, 0],
      "Yes": [1, 1, 0, 0, 0, 0],
      "No": [1, 1, 0, 1, 0, 0]
    },
    {
      "Practice of Vegetarian Diet": [1, 0, 1, 0, 0, 0],
      "Yes": [1, 1, 0, 0, 0, 0],
      "No": [1, 1, 0, 1, 0, 0],
      "Flexitarian": [1, 1, 0, 1, 0, 0],
      "Always Vegetarian": [1, 1, 1, 0, 0, 0]
    }
  ]
}
"""),
    "treemap" : dedent("""  
        - **추출한 데이터는 일본어로 출력하라 ** 
        - treemap JSON 구조는 반드시 아래 형식만 허용한다.
        - series 내부에는 오직 "name", "categories" 필드만 존재해야 한다.
        - categories는 반드시 리스트이며, 각 항목은 { "name": string, "value": number } 구조만 허용.
        - series 안에 하나의 묶음 = 파이차트 하나를 뜻함. 
        
        - legend는 series와 categories의 이름을 key로 가지고,
          값은 서로 겹치지 않는 2×3 패턴 배열이다.
        - legend는 절대 겹치지 않게 하고, 반드시 아래 우선순위를 따른다.
          [1,0,0,0,0,0] [1,0,1,0,0,0] [1,1,0,0,0,0] [1,1,0,1,0,0]
          [1,0,0,1,0,0] [1,1,1,0,0,0] [1,0,1,1,0,0] [1,1,1,1,0,0] [0,1,1,0,0,0] ...

        - treemap이 여러 개 있어 보여도 series 배열에 각각 하나의 series로 나열한다.
        - chart 객체는 항상 단 하나만 생성한다. 여러 chart 객체 리스트 절대 금지.

        - 출력 예시는 아래의 구조를 항상 따라야 한다: 절대 이상한 값이나 구조를 추가하거나 삭제하지말것. 무조건 아래의 구조에서 다른 구조를 추가하려고 하지 말 것. 
                       
                       
{
  "chart_type": {
    "type": "treemap"
  },
  "series": [
    {
      "name": "Breakfast",
      "categories": [
        { "name": "Waffle", "value": 18 },
        { "name": "Egg", "value": 15 },
        { "name": "Pancake", "value": 12 }
      ]
    },
    {
      "name": "Lunch",
      "categories": [
        { "name": "Salad", "value": 12 },
        { "name": "Castella", "value": 15 }
      ]
    }
  ],
  "legend": [
    {
      "Breakfast": [1, 0, 0, 0, 0, 0],
      "Waffle": [1, 1, 0, 0, 0, 0],
      "Egg": [1, 1, 0, 1, 0, 0],
      "Pancake": [1, 0, 0, 1, 0, 0]
    },
    {
      "Lunch": [1, 0, 1, 0, 0, 0],
      "Salad": [1, 1, 1, 0, 0, 0],
      "Castella": [1, 0, 1, 1, 0, 0]
    }
  ]
}
"""),
    "violin" : dedent("""{
          "chart_type": { "type": "violin" }, #
          "series": ["Yes", "No"],
          "categories": ["Thur", "Fri"],
          "data": {
            "Thur": {
              "Yes": {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]},
              "No":  {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]}
            },
            "Fri": {
              "Yes": {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]},
              "No":  {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]}
            }
          },
          "axes": { "Y": { "range": [0, 30], "interval": 5 } },
          "legend": {
            "Yes":  [1,0,0,0,0,0],
            "No":  [1,0,1,0,0,0] 
          }
        }
"""),
    "boxplot" : dedent("""{
          "chart_type": { "type": "boxplot" },
          "series": ["Yes", "No"],
          "categories": ["Thur", "Fri"],
          "data": {
            "Thur": {
              "Yes": {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]},
              "No":  {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]}
            },
            "Fri": {
              "Yes": {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]},
              "No":  {"min":0,"Q1":10,"median":20,"Q3":30,"max":50,"outliers":[""]}
            }
          },
          "axes": { "Y": { "range": [0, 30], "interval": 5 } },
          "legend": {
            "Yes":  [1,0,0,0,0,0],
            "No":  [1,0,1,0,0,0] 
          }
        }
"""),
    "mixed" : dedent("""{
          "chart_type": {
            "type": "mixed",
            "components": [
              { "type": "bar",  "series": ["Min", "Max"], "y_axis": "left"  },
              { "type": "line", "series": ["Median"],     "y_axis": "right" }
            ]
          },
          "series": ["Min", "Max", "Median"],
          "categories": ["Product A","Product B","Product C","Product D","Product E"],
          "data": {
            "Min":    [5,10,7,12,6],
            "Max":    [95,100,90,97,93],
            "Median": [75,85,65,80,70]
          },
          "y_axes": {
            "left":  { "label": "Review Score Range", "range": [0,100], "interval": 5  },
            "right": { "label": "Median Review Score", "range": [50,100], "interval": 5  }
          },
          "legend": {
            "Min":    [1,0,0,0,0,0],
            "Max":    [1,0,1,0,0,0] ,
            "Median": [1,1,0,0,0,0] 
          }
        }
"""),
    "scatter" : dedent("""{
          "chart_type": { "type": "scatter" },
          "series": ["data1", "data2", "data3"],
          "categories": [""],
          "data": { "1": [""] },  //산점도는 데이터를 저장하지 않음
          "legend": {
            "data1":  [1,0,0,0,0,0],
            "data2": [1,0,1,0,0,0],
            "data3": [1,1,0,0,0,0] 
          }
        }"""),
}

def get_dict(name : str):
    return EXAMPLE_BLOCK_DCIT.get(name, None)

def get_image_descript():
    return """
    **回答は必ず日本語で行うこと。**  
例が他の言語で書かれていても、必ず日本語で答えること。

本システムは、チャートを音声で説明するTTS用の要約を作成する役割を持つ。  
以下のような構造化された形式で、**日本語のみで出力**しなければならない。

<チャートのタイトルおよびチャートの種類>  
チャートのタイトルとチャートの種類を一文で説明する。  
チャートの種類は以下のいずれかで述べること：  
[棒グラフ、折れ線グラフ、円グラフ、複合グラフ、二重棒グラフ、ツリーマップ、散布図、バイオリンプロット、箱ひげ図]

<Y軸の情報>  
   - 左軸：ラベル、値の範囲、間隔  
   - 右軸：ラベル、値の範囲、間隔  

<X軸の情報>  
   - ラベルの説明（例：月、年度、カテゴリなど）  
   - 値またはカテゴリの一覧（順番通りに列挙）

<データ系列の情報>  
   - 各データ系列の名称  
   - 値：X軸のカテゴリに対応する値を順番通りに列挙すること

<総合要約>  
   - 全体としてどのような傾向が見られるかを簡潔に説明する（上昇・下降・変動など）

出力は**必ず同じ構造**に従うこと。  
詳細な視覚的描写は行わず、数値・カテゴリ・変化を簡潔に説明する内容のみ含める。

"""

def get_chart_type_template_image():
    block = get_dict("EXAMPLE_STRUCT")
    system_prompt = (
      f"""
      다음 이미지는 차트입니다. 차트가 bar, line, pie, scatter, box, violin, treemap, mixed 인지 구별한 후 다음 형식에 따라 분석 결과를 JSON으로 출력하세요. (mixed는 2개 이상의 차트가 결합된 형태입니다.)
      데이터 값이 명시되어 있지 않더라도, 색상 크기나 길이, 위치를 기반으로 유추하여 `data` 항목을 *정확하게* 채워주세요.
      - *, -  이런 부가적인 특수문자 포함 금지. 오직 한글과 마침표만 사용.

      {block}
      """
    )

    return  system_prompt

def get_chart_type_template_text():
    block = get_dict("EXAMPLE_STRUCT")
    system_prompt = (
        f"""
        사용자는 어떠한 특정 데이터를 텍스트 혹은 로우 데이터로 제공합니다. 사용자가 제공하는 정보를 기반으로 차트 데이터를 생성합니다. 
        차트가 bar, line, pie, scatter, boxplot, violin, treemap, mixed 인지 구별한 후 다음 형식에 따라 분석 결과를 JSON으로 출력하세요. (mixed는 2개 이상의 차트가 결합된 형태입니다.) 또한, 로우데이터는 그것 차체로 넘기지 않고, 중요한 값을 기반으로 평균을 내서 사용해도 됩니다. 사용자가 원하는 차트 종류로 제공하세요. 
        만약 사용자가 어떠한 차트의 종류라고 명시 하지 않았다면 데이터의 특성을 기반으로 사람이 보기 쉬운 차트형식으로 제공해주세요.  
        차트 데이터는 series와 categories를 합쳐서 10개가 넘어가지 않도록 합니다. 이를 위해서 값을 평균을 내는 등, 형태를 바꿔서 사용하세요. 
        사용자가 제시한 데이터를 기반으로 `data` 항목을 *정확하게* 채워주세요.
        - *, -  이런 부가적인 특수문자 포함 금지. 오직 한글과 마침표만 사용.

        {block}
        """
    )

    return  system_prompt

def get_scatter_QA(rgb: str):
    system_prompt = (
        f'''
        대답은 무조건 일본어로 한다. 예시는 다른 언어더라도 무조건 일본어로 대답한다. 

        모든 입력은 차트(그래프) 이미지와 관련된 질문입니다. 아래 지침을 반드시 따르세요.

        ** 차트에 있는 값들의 색들은 다음과 같습니다. {rgb} 이 색들에 해당하는 라벨을 잘 생각하고 답하세요. **

        1. 모든 답변은 반드시 한국어로 작성해야 합니다.
        2. 답변은 간단하고 명확하게 서술하세요.
        3. 질문의 끝부분에 제공되는 RGB 값 목록 중, 당신이 답변에서 언급한 데이터 항목(레이블)에 해당하는 RGB 값만 답변 끝에 붙이세요.
        4. 질문에서 언급되지 않은 항목의 RGB 값은 절대 붙이지 마세요.
        5. RGB 값은 항목명 옆에 색상명을 쓰지 말고 (R, G, B) 형태로 숫자만 괄호로 표기하세요.
        6. 답변을 마친 후, RGB 값들을 모두 나열하고, 마지막에 차트 유형을 'bar', 'line', 'pie', 'scatter', 'None' 중 하나로 표기하세요.
        7. RGB 값은 반드시 질문에서 제공된 목록 중에서만 선택해야 하며, 목록에 없는 색상은 절대 사용하지 마세요.

        예시:
        질문: "setosa와 versicolor의 분포를 알려줘.(0, 0, 255) (255, 165, 0) (0, 255, 0)"
        답변: "이 차트는 setosa와 versicolor의 점 분포를 보여줍니다. setosa는 아래쪽에 몰려 있고 versicolor는 위쪽에 분포합니다." (0, 0, 255), (255, 165, 0) scatter

        질문: "자전거에 대해서 알려줘. "
        답변: "자전거는 24%입니다." (123, 34, 59) scatter
        '''
    )
    
    return system_prompt

def get_finger_positions(chart_json): 
    
    system_prompt = (
        f'''
        대답은 무조건 일본어로 한다. 예시는 다른 언어더라도 무조건 일본어로 대답한다. 


        당신의 임무는 두 장의 이미지를 비교하여,
        (1) 촉각그래픽 이미지에서 표시된 빨간 점이 어떤 데이터 구조(박스, 바이올린, 트리맵, 파이차트, 선 그래프, 막대그래프, 막대+선 그래프, 산점도)에 속하는지,
        (2) 원본 차트의 어떤 데이터 요소와 정확히 대응하는지를 분석하는 것이다.

        분석 절차
        1) 촉각그래픽 구조 해석

        빨간 점이 흰색 데이터 블록 안에 명확히 포함되면 → 반드시 그 데이터에 귀속한다.

        촉각그래픽은 원본 차트의 시각적 요소를 단순화한 것이다.
        특히 작은 데이터 조각이 실제보다 과장되게 표현될 수 있으므로, 단순한 면적 비교보다는 위치 중심의 해석이 중요하다.

        파이차트의 경우, 각 조각의 상대적 위치(시계 방향 각도), 분할된 조각의 개수 및 배열 구조를 기준으로 조각 자체를 파악하고,
        빨간 점이 해당 조각 내부에 있는지를 판별한다.
        즉, 빨간 점이 조각 중심에 가까운지를 보기보다, 어떤 조각에 귀속되는지를 먼저 파악해야 한다.

        점이 위치한 영역이 어떤 차트 구조에 해당하는지 식별하라:
        • 박스플롯: 박스(사분위), 중앙선(중앙값), 위/아래 수염, 이상치
        • 바이올린: 밀도 곡선 내부, 중앙선/사분위
        • 트리맵: 사각형 블록(계층)
        • 파이: 조각 자체 기준으로 특정 (조각 수, 시계 방향, 위치 구조 기반 판단)
        • 선 그래프: 선 위 마커(작은 사각형 2×3 점)
        • 막대 그래프: 특정 막대(범례 색 포함)
        • 막대+선 그래프: 막대와 선을 구분(선은 2×3 마커)
        • 산점도: 개별 점

        점이 데이터 구조에 명확히 속하지 않고 검정색(배경) 빈 공간이면 결과를 산출하지 말고 아래 출력 규칙의 문구만 반환한다.

        단, 점이 경계에 걸렸더라도 조각 또는 마커의 구조상 명백히 귀속된다고 판단될 경우 해당 데이터로 귀속한다.

        2) 원본 차트 데이터와 대응

        식별된 구조를 원본 차트의 실제 데이터와 연결하라.

        점이 둘 이상의 범례 데이터에 동시에 속하고 값이 동일하면 모든 범례를 말하고 문장 끝에 “~와 ~의 값이 동일합니다.”를 덧붙인다.

        어느 데이터에도 속하지 않으면 아무 설명 없이 출력 규칙의 문구만 반환한다.

        출력 규칙 (아주 중요)

        반드시 한 줄로 출력한다.

        데이터가 있으면 사용자의 질문에 대한 답을 한다. 

        완전히 빈 영역이면: "정확한 위치를 알려주세요"라고 한다. 

        위 형식을 제외한 어떠한 설명도 덧붙이지 마라.

        [원본 차트 데이터]
        {chart_json}

        ''')
    return system_prompt

def get_general_chart_QA(chartQA_data, chart_type):
    
    QA_prompt = f"""

    대답은 무조건 일본어로 한다. 예시는 다른 언어더라도 무조건 일본어로 대답한다. 
    사용자의 질문에 일본어로 단답으로 답변.     다음 규칙을 반드시 지킵니다.


    <전환 가능 표> 
    만일 아래 가능한 표 외에 다른 그래프로 변환 요청이오면, 불가하다는 말과 함께 가능한 그래프 종류를 설명해라.

    "막대 그래프": ["선 그래프"],
    "선 그래프": ["막대 그래프"],
    "파이 그래프": ["트리맵"],
    "트리맵": ["파이 그래프"],
    "박스플롯": ["바이올린 차트"],
    "바이올린 차트": ["박스플롯"],
    "산점도": [],
    "혼합 차트": []


    [절대 금지]
    - "이미지를 수정/분석할 수 없다", "데이터를 제공해 달라" 등 사과/요청/거절 문구 금지.
    - 데이터를 나열하며 "이걸로 그리라" 안내 금지.
    - *, -  이런 부가적인 특수문자 포함 금지. 오직 한글과 마침표만 사용.

    [판정 규칙]
    1) 사용자가 "~보여줘/~그려줘/"라고 하면, 차트 타입과 관계없이 현재 데이터에 있는 값 중 사용자가 지정한 시리즈(범례와 유사한 이름 포함)의 값을 나열한다. 이때 혼합 차트라 하더라도 지정한 시리즈만 필터링하여 값 나열로 응답한다.
    2) 사용자가 <현재 차트타입>과 다른 타입의 그래프 변환을 요청하면 "Chart conversion"을 시도한다. 

    4) 사용자는 보통 사진 속 데이터 이름을 기반으로 질문하거나, 전체 차트에 대한 설명을 요구한다.사용자가 사진 속 범례에 있는 데이터를 그대로 말하지 않을 수 있다. "Price"를 "가격"으로 번역할 수도 있고, "운동률에 따른 심박수"를 "운동률"로 줄여서 부를 수도 있다. 정확한 범례는 아니지만 유사한 범례가 있다면 그 값이라 생각하고 답하여라. 
      

    [응답 형식]
    - When only values are possible (e.g., "Tell me the largest value in July", "Explain the rest of the data except for Jeong Daeman's data"): Write a simple one-line description (e.g., "July maximum value 900", "Seo Taewoong value: 64, Kang Baekho value: 32").
    - 차트 변환만 필요한 경우:  "タイプAのチャートをタイプBのチャートに変換します。" 출력

    <차트 데이터>
    {chartQA_data}

    <현재 차트타입>
    {chart_type}
    """
    return QA_prompt

def get_conversion(text, chart_type, chart_json, convertTarget):
    block = get_dict(convertTarget)

    system_prompt = (f"""
          '{text}' 사용자는 차트 변환을 요청했습니다. 아래 차트 데이터 형식을 참고하여, 원본 데이터의 값을 변형하지 않고, [변환시킬 차트 구조] 형식으로만 구조를 변환하여  JSON으로 출력하세요.

          [중요 규칙: legend 불변]
          - legend는 원본의 값을 '그대로' 복사합니다. 요소 값을 수정/추정/패딩 금지.
          - series를 필터링한 경우에도, 남긴 series의 legend는 원본의 해당 series 값을 '그대로' 사용합니다.


          [원본 데이터]
          {chart_json}

          [변환시킬 차트 구조] 이 구조에 맞게 차트 타입부터 구조를 변환하세요. 
          {block}

          """)
    return system_prompt

def get_highlight(chartQA_data, text):
    
    system_prompt = f"""
      너는 차트 JSON을 분석하여 사용자의 질문 조건에 맞는 하이라이트 대상만 JSON으로 출력하는 역할을 한다.

      📌 기본 규칙
      1) 현재 JSON의 series, category 값은 정확히 동일하게 사용(대소문자·띄어쓰기·순서 유지).
      2) series와 category 혼동 금지.
      3) data 값은 변경 불가.
      4) 조건이 불명확하거나 일치하는 데이터 없음 → {{"highlight_mode": "all"}}만 반환.
      5) 출력은 하나의 JSON 객체로만, 설명·주석·불필요한 텍스트 금지.
      6) 질문에서 요청한 조건과 일치하는 모든 항목을 누락 없이 포함.
        - 특정 값을 말하고 있다면 현재 데이터에서 해당 값을 모두 찾음.
        - 최대/최소 등 계산이 필요한 조건은 현재 데이터로 계산 후, 해당되는 (series, category) 조합을 모두 "custom" 모드로 반환.
        - 여러 개의 값(다중 series/category 조합)이 포함된 경우 → 반드시 "custom" 모드로 처리. 

    📌 출력 포맷 (허용 모드)
    - 특정 series 전체:
      {{"highlight_mode": "series", "custom_indices": [{{"series": "<값>"}}, ...]}}
    - 특정 category 전체:
      {{"highlight_mode": "category", "categories": ["<값>", ...]}}
    - 특정 series & category 조합(여러 개 가능):
      {{"highlight_mode": "custom", "custom_indices": [{{"series": "<값>", "category": "<값>"}}, ...]}}
    - 조건 불명확/일치 없음:
      {{"highlight_mode": "all"}}

    📌 처리 절차
    1) USER의 말에서 조건을 해석.
    2) 현재 JSON 데이터에서 해당 조건에 맞는 series/category 조합을 전부 찾음. 절대 현재 데이터에서 값을 유추하거나 지어내지 말 것. series/category를 혼돈하지 말 것.
    3) 질문이 series 전체만 지칭 → "series" 모드 사용.
    4) 질문이 category 전체만 지칭(복수 가능) → "category" 모드 사용.
    5) 특정 series와 category 조합(복수 포함), 또는 최대/최소/필터 등 계산 조건 → 반드시 "custom" 모드로 반환.
    6) 조건이 불명확하거나 매칭 없음 → "all" 반환.
    7) 출력 JSON은 단일 객체, 키 순서 유지, 누락·추가 없이 반환.

      현재 데이터:
      {chartQA_data}
      사용자의 질문:
      {text}
    """
    return system_prompt

