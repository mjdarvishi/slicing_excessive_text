import nltk

# Example usage:
input_text = '''
Health is a multifaceted and dynamic concept encompassing the overall well-being of an individual, not merely the absence of disease. Achieving and maintaining good health involves a holistic approach that integrates physical, mental, and social dimensions. It is a state of complete physical, mental, and social well-being and not merely the absence of infirmity.

Physical health is a cornerstone of overall well-being. Regular physical activity, a balanced diet, and sufficient sleep contribute to optimal physical health. Engaging in regular exercise helps maintain a healthy weight, improves cardiovascular health, and enhances muscular strength and flexibility. A nutrient-rich diet, rich in fruits, vegetables, whole grains, and lean proteins, provides essential vitamins and minerals necessary for bodily functions.

Mental health is equally crucial in the pursuit of overall well-being. It encompasses emotional, psychological, and social aspects of an individual's life. Maintaining good mental health involves managing stress, cultivating positive relationships, and developing coping mechanisms for life's challenges. Practices such as mindfulness, meditation, and seeking professional support when needed contribute to mental well-being.

Social health recognizes the importance of interpersonal relationships and a supportive social environment. Strong social connections, effective communication, and a sense of belonging positively impact mental and emotional health. Engaging in community activities, spending quality time with friends and family, and fostering meaningful connections promote social well-being.

Preventive healthcare plays a pivotal role in maintaining good health and preventing the onset of diseases. Regular health check-ups, vaccinations, and screenings help identify potential health risks early, allowing for timely intervention. Adopting a proactive approach to health, including maintaining a healthy lifestyle, can significantly reduce the risk of chronic diseases such as diabetes, heart disease, and certain cancers.

Health education and awareness are essential components of promoting well-being. Empowering individuals with knowledge about healthy lifestyle choices, nutrition, and the importance of regular exercise enables them to make informed decisions about their health. Public health campaigns, educational programs, and accessible healthcare information contribute to creating a health-conscious society.

Environmental factors also influence health outcomes. Clean air, safe drinking water, and access to green spaces contribute to a healthy living environment. Addressing environmental challenges such as pollution and promoting sustainable practices contribute not only to individual health but also to the well-being of communities and the planet.

Ultimately, achieving and maintaining good health is a lifelong journey that requires commitment, self-awareness, and a proactive approach to well-being. It involves recognizing the interconnectedness of physical, mental, and social aspects of health and making choices that align with a holistic understanding of overall well-being. By prioritizing health and adopting a comprehensive approach, individuals can enhance their quality of

'''

standard_size = 2048
def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download("stopwords")
        nltk.download('wordnet')