import React from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  DATASET_OPTIONS,
  SUBSET_OPTIONS,
  METRIC_OPTIONS,
} from "../constants/constants";
import { setDataSet, setSubset, setMetric } from "../redux/slices/imagesSlice";
import "../styles/SelectContainer.scss";

const SelectContainer = () => {
  const { dataSet, subset, metric } = useSelector(
    (state) => state.imagesReducer
  );
  const dispatch = useDispatch();
  return (
    <div className="SelectContainer">
      <select
        name="dataSet"
        title="dataSet"
        value={dataSet}
        onChange={(e) => {
          dispatch(setDataSet(e.target.value));
        }}
      >
        <option value="" disabled hidden>
          Select dataSet
        </option>
        {DATASET_OPTIONS.map((item, key) => (
          <option value={item} key={key}>
            {item}
          </option>
        ))}
      </select>

      <select
        name="subset"
        title="subset"
        value={subset}
        onChange={(e) => {
          dispatch(setSubset(e.target.value));
        }}
      >
        <option value="" disabled hidden>
          Select model
        </option>
        {SUBSET_OPTIONS.map((item, key) => (
          <option value={item} key={key}>
            {item}
          </option>
        ))}
      </select>

      <select
        name="metric"
        title="metric"
        value={metric}
        onChange={(e) => {
          dispatch(setMetric(e.target.value));
        }}
      >
        <option value="" disabled hidden>
          Select dataType
        </option>
        {METRIC_OPTIONS.map((item, key) => (
          <option value={item} key={key}>
            {item}
          </option>
        ))}
      </select>
    </div>
  );
};

export default SelectContainer;
