import tf from "@tensorflow/tfjs-node"
import fs from "fs"

const MODEL_PATH = "./models/basic_model"

const CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz".split("")
const ENCODING_SIZE = CHARACTER_SET.length + 1
const MAX_WORD_SIZE = 15
const TIME_STEPS = 3

const TAGS = ['WH', 'ADV', 'MOD', 'PRON', 'VERB', 'TO', 'DT', 'ADJ', 'NOUN', 'PREP', 'CONJ', 'NUMB', 'PART', 'AUX']
const TAG_ENCODING_SIZE = TAGS.length + 1

const MAX_BATCH = 100


const oneHotEncodeToken = (token) => {
  const characters = token.toLowerCase().split("")
  const characterIndices = characters.map(c => CHARACTER_SET.indexOf(c) + 1)
  characterIndices.length = MAX_WORD_SIZE
  const characterTensor = tf.tensor1d(characterIndices, "int32")
  const encodedCharacters = tf.oneHot(characterTensor, ENCODING_SIZE)
  return encodedCharacters.reshape([MAX_WORD_SIZE * ENCODING_SIZE])
}

const encodeSentence = (sentence) => {
  const tokens = sentence.split(/\s+/)
  return tf.stack(tokens.map(oneHotEncodeToken))
}

const padEncodedSentence = (encodedSentence) => {
  const zeros = encodeSentence("").tile([TIME_STEPS - 1, 1])
  return zeros.concat(encodedSentence)
}

const encodeAndPadSentence = (sentence) => padEncodedSentence(encodeSentence(sentence))

const buildSentenceInputs = (sentence) => {
  const encodedSentence = encodeAndPadSentence(sentence)
  const inputs = []
  for (let i = 0; i < encodedSentence.shape[0] - TIME_STEPS + 1; i++) {
    inputs[i] = encodedSentence.slice(i, TIME_STEPS)
  }
  return tf.stack(inputs)
}

const oneHotEncodeTag = (tag) => {
  const tag_index = TAGS.indexOf(tag) + 1
  return tf.oneHot(tag_index, TAG_ENCODING_SIZE)
}

const encodeTags = (tags) => {
  return tf.stack(tags.map(oneHotEncodeTag))
}

const padEncodedTags = (encodedTags) => {
  const zeros = encodeTags([""]).tile([TIME_STEPS - 1, 1])
  return zeros.concat(encodedTags)
}

const encodeAndPadTags = (tags) => padEncodedTags(encodeTags(tags))

const buildTagOutputs = (tags) => {
  const encodedTags = encodeAndPadTags(tags)
  const outputs = []
  for (let i = 0; i < encodedTags.shape[0] - TIME_STEPS + 1; i++) {
    outputs[i] = encodedTags.slice(i, TIME_STEPS)
  }
  return tf.stack(outputs)
}

const decodeTags = (predictedTensor) => {
  const predictedArr = predictedTensor.arraySync()
  return predictedArr.map((prediction) => {
    const lastPrediction = prediction[TIME_STEPS - 1]
    
    let max = 0.0
    let tag_index = lastPrediction.reduce((acc, v, i) => {
      let max_index = acc
      if (v > max && i > 0) {
        max = v
        max_index = i
      }
      return max_index
    }, 0)

    return tag_index > 0 ? TAGS[tag_index-1] : "NO TAG"
  })
}

let model = null
if (fs.existsSync(MODEL_PATH)) {
  model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`)
} else {
  const input = tf.input({shape: [TIME_STEPS, MAX_WORD_SIZE * ENCODING_SIZE]})
  
  const rnn = tf.layers.simpleRNN({units: TAG_ENCODING_SIZE, returnSequences: true});
  const output = rnn.apply(input);
  
  model = tf.model({inputs: input, outputs: output})
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
  
  model.summary()
  
  const data = JSON.parse(fs.readFileSync("data/data.json"))
  
  for (let i = 0; i < data.length; i += MAX_BATCH) {
    const batch = data.slice(i, i + MAX_BATCH)
    let xs, ys
  
    batch.forEach(({sentence, tags}, j) => {
      let sentenceInputs = buildSentenceInputs(sentence)
      let tagOutputs = buildTagOutputs(tags)
  
      if (j == 0) {
        xs = sentenceInputs
        ys = tagOutputs
      } else if (sentenceInputs.shape[0] === tagOutputs.shape[0]) {
        xs = xs.concat(sentenceInputs)
        ys = ys.concat(tagOutputs)
      }
    })
  
    await model.fit(xs, ys, {
      epochs: 100,
      verbose: 0,
      callbacks: {
        onTrainEnd: (logs) => { console.log(`Ended training #${i}-${i + batch.length - 1}`) }
      }
    })
  }

  await model.save(`file://${MODEL_PATH}`)
}

const testModel = (model, sentence) => {
  console.log(`Testing sentence "${sentence}"`)
  const sentenceInputs = buildSentenceInputs(sentence)
  const output = model.predict(sentenceInputs)
  const outputTags = decodeTags(output)
  const tokens = sentence.split(/\s+/)

  for (let i = 0; i < tokens.length; i++) {
    console.log(tokens[i], "=>", outputTags[i])
  }
}

testModel(model, "The man is a car")


