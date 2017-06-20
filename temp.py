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