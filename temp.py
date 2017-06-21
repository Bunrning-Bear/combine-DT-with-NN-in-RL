for val in values:
    # Create a subtree for the current value under the "best" field
    # [question] for countinus attribute, should we regard it as discrete points. means create a node for each points?
    if is_continuous:
        selected_data = [r for r in data if r[best] <= val]
    else:
        selected_data = [r for r in data if r[best] == val]
    subtree = create_decision_tree(
        selected_data,
        [attr for attr in attributes if attr != best],
        class_attr,
        fitness_func,
        split_attr=best,
        split_val=val,
        wrapper=wrapper,
        current_deep=current_deep+1)

    # Add the new subtree to the empty dictionary object in our new
    # tree/node we just created.
    if isinstance(subtree, Node):
        node._branches[val] = subtree
    elif isinstance(subtree, (CDist, DDist)):
        node.set_leaf_dist(attr_value=val, dist=subtree)
        #             wrapper.leaf_count += 1
        #  self.leaf_list
    else:
        raise Exception("Unknown subtree type: %s" % (type(subtree),))


    def set_leaf_dist(self, attr_value, dist):
        """
        Sets the probability distribution at a leaf node.
        """
        assert self.attr_name
        assert self.tree.data.is_valid(self.attr_name, attr_value), \
            "Value %s is invalid for attribute %s." \
                % (attr_value, self.attr_name)
        if self.is_continuous_class:
            assert isinstance(dist, CDist)
            assert self.attr_name
            self._attr_value_cdist[self.attr_name][attr_value] = dist.copy()
#            self.n += dist.count
        else:
            assert isinstance(dist, DDist)
            # {attr_name:{attr_value:count}}
            self._attr_value_counts[self.attr_name][attr_value] += 1
            # {attr_name:total}
            self._attr_value_count_totals[self.attr_name] += 1
            # {attr_name:{attr_value:{class_value:count}}}
            for cls_value, cls_count in iteritems(dist.counts):
                self._attr_class_value_counts[self.attr_name][attr_value] \
                    [cls_value] += cls_count


    def set_leaf_dist(self, attr_value, dist):
        """
        [delete this function]
        Sets the probability distribution at a leaf node.
        """
        assert self.attr_name
        assert self.tree.data.is_valid(self.attr_name, attr_value), \
            "Value %s is invalid for attribute %s." \
                % (attr_value, self.attr_name)
        if self.is_continuous_class:
            assert isinstance(dist, CDist)
            assert self.attr_name
            # dist.copy() is the mean of class value
            self._attr_value_cdist[self.attr_name][attr_value] = dist.copy()
#            self.n += dist.count
        else:
            assert isinstance(dist, DDist)
            # {attr_name:{attr_value:count}}
            self._attr_value_counts[self.attr_name][attr_value] += 1
            # {attr_name:total}
            self._attr_value_count_totals[self.attr_name] += 1
            # {attr_name:{attr_value:{class_value:count}}}
            for cls_value, cls_count in iteritems(dist.counts):
                self._attr_class_value_counts[self.attr_name][attr_value] \
                    [cls_value] += cls_count




INFO:root:best attribute is workclass
INFO:root:the values of this best attributes:[set([0, 1, 3, 4, 5])]
INFO:root:now we choose value (0),from [workclass]
INFO:root:best attribute is age
INFO:root:{'fnlwgt': [28887.0, 544091.0], 'age': [19.0, 59.0], 'education': [0, 13], 'workclass': [0, 5]}
INFO:root:the values of this best attributes:[[34, 9999999999]]
INFO:root:now we choose value (34),from [age]
    {'fnlwgt': 338409.0, 'age': 28.0, 'education': 0, 'workclass': 0, 'cls': 1}
    {'fnlwgt': 45781.0, 'age': 31.0, 'education': 10, 'workclass': 0, 'cls': 0}
    {'fnlwgt': 122272.0, 'age': 23.0, 'education': 0, 'workclass': 0, 'cls': 1}
    {'fnlwgt': 205019.0, 'age': 32.0, 'education': 5, 'workclass': 0, 'cls': 1}
INFO:root:now we choose value (9999999999),from [age]
    {'fnlwgt': 215646.0, 'age': 38.0, 'education': 3, 'workclass': 0, 'cls': 1}
    {'fnlwgt': 234721.0, 'age': 53.0, 'education': 2, 'workclass': 0, 'cls': 1}
    {'fnlwgt': 284582.0, 'age': 37.0, 'education': 10, 'workclass': 0, 'cls': 1}
    {'fnlwgt': 160187.0, 'age': 49.0, 'education': 7, 'workclass': 0, 'cls': 1}
    {'fnlwgt': 159449.0, 'age': 42.0, 'education': 0, 'workclass': 0, 'cls': 0}
    {'fnlwgt': 280464.0, 'age': 37.0, 'education': 1, 'workclass': 0, 'cls': 0}
    {'fnlwgt': 245487.0, 'age': 34.0, 'education': 8, 'workclass': 0, 'cls': 1}
INFO:root:now we choose value (1),from [workclass]
INFO:root:best attribute is education
INFO:root:the values of this best attributes:[set([0, 1, 2, 3, 5, 7, 8, 10, 13])]
INFO:root:now we choose value (0),from [education]
    {'fnlwgt': 83311.0, 'age': 50.0, 'education': 0, 'workclass': 1, 'cls': 1}
INFO:root:now we choose value (1),from [education]
INFO:root:now we choose value (2),from [education]
INFO:root:now we choose value (3),from [education]
    {'fnlwgt': 209642.0, 'age': 52.0, 'education': 3, 'workclass': 1, 'cls': 0}
INFO:root:now we choose value (5),from [education]
INFO:root:now we choose value (7),from [education]
INFO:root:now we choose value (8),from [education]
INFO:root:now we choose value (10),from [education]
INFO:root:now we choose value (13),from [education]
INFO:root:now we choose value (3),from [workclass]
INFO:root:best attribute is education
INFO:root:the values of this best attributes:[set([0, 1, 2, 3, 5, 7, 8, 10, 13])]
INFO:root:now we choose value (0),from [education]
INFO:root:now we choose value (1),from [education]
INFO:root:now we choose value (2),from [education]
INFO:root:now we choose value (3),from [education]
INFO:root:now we choose value (5),from [education]
INFO:root:now we choose value (7),from [education]
INFO:root:now we choose value (8),from [education]
INFO:root:now we choose value (10),from [education]
INFO:root:now we choose value (13),from [education]
INFO:root:now we choose value (4),from [workclass]
INFO:root:best attribute is fnlwgt
INFO:root:{'fnlwgt': [28887.0, 544091.0], 'age': [19.0, 59.0], 'education': [0, 13], 'workclass': [0, 5]}
INFO:root:the values of this best attributes:[[186791, 9999999999]]
INFO:root:now we choose value (186791),from [fnlwgt]
INFO:root:now we choose value (9999999999),from [fnlwgt]
INFO:root:now we choose value (5),from [workclass]
INFO:root:best attribute is education
INFO:root:the values of this best attributes:[set([0, 1, 2, 3, 5, 7, 8, 10, 13])]
INFO:root:now we choose value (0),from [education]
    {'fnlwgt': 141297.0, 'age': 30.0, 'education': 0, 'workclass': 5, 'cls': 0}
INFO:root:now we choose value (1),from [education]
INFO:root:now we choose value (2),from [education]
INFO:root:now we choose value (3),from [education]
INFO:root:now we choose value (5),from [education]
INFO:root:now we choose value (7),from [education]
INFO:root:now we choose value (8),from [education]
INFO:root:now we choose value (10),from [education]
INFO:root:now we choose value (13),from [education]