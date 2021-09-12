import {getModel, runTests} from "./modelBuilder.js"
import {writeResults} from "./rank.js"

// const RNN_SIZES = [10, 30, 60, 90]
// const TIME_STEPS = [10, 15, 20, 25]
// const EPOCHS = [100, 200, 300]

const RNN_SIZES = [60]
const TIME_STEPS = [15, 20, 25]
const EPOCHS = [100, 200, 300]

for (let rnnSize of RNN_SIZES) {
  for (let timeSteps of TIME_STEPS) {
    for (let epochs of EPOCHS) {
      let [model, modelPath] = await getModel({rnnSize, timeSteps, epochs})
      runTests(model, modelPath)
    }
  }
}

writeResults()
