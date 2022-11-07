import { createSlice } from "@reduxjs/toolkit";

export const imagesSlice = createSlice({
  name: "imagesReducer",
  initialState: {
    dataSet: "",
    subset: "",
    metric: "",
  },
  reducers: {
    setDataSet(state, action) {
      state.dataSet = action.payload;
    },
    setSubset(state, action) {
      state.subset = action.payload;
    },
    setMetric(state, action) {
      state.metric = action.payload;
    },
  },
});

export const { setDataSet, setSubset, setMetric } = imagesSlice.actions;

export function getDataSetImgLink(dataSet) {
  const isOptionsSelected = dataSet !== "";
  return isOptionsSelected ? `${dataSet}` : "";
}

export function getInputImgLink(props) {
  const isOptionsSelected = !Object.values(props).includes("");

  const { dataSet, subset, metric } = props;

  return isOptionsSelected ? `${dataSet}/${subset}/${metric}` : "";
}

export default imagesSlice.reducer;
